import os
import time
import json
from typing import Generator, List, Dict, Any, Optional

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm
from langchain_core.messages import HumanMessage

from agent import build_graph

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
FALLBACK_SPACE_ID = "ecandeloro/hf_agent_gaia_30"  # change if you fork

# temp directories setup
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp")  # Default temp directory
QUESTIONS_FILES_DIR = os.path.join(TEMP_DIR, "questions_files")
OUTPUT_GAIA_DIR = os.path.join(TEMP_DIR, "output_gaia")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(QUESTIONS_FILES_DIR, exist_ok=True)
os.makedirs(OUTPUT_GAIA_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Agent wrapper
# ─────────────────────────────────────────────────────────────────────────────


class BasicAgent:
    """Very small shim around the langgraph returned by `build_graph()`."""

    TAG = "[FINAL ANSWER]"

    def __init__(self) -> None:
        print("⏳  Initialising BasicAgent …")
        self.graph = build_graph()
        print("✅  BasicAgent ready!")

    def __call__(self, question: str, input_file: Optional[str]) -> str:
        print(f"⚙️   Processing: {question[:60]}…")
        msgs = [HumanMessage(content=question)]
        chain_out = self.graph.invoke(
            {"messages": msgs, "input_file": input_file})

        raw = chain_out["messages"][-1].content

        # ── keep only what follows the *last* “[FINAL ANSWER]” tag ──
        idx = raw.rfind(self.TAG)
        answer = raw[idx + len(self.TAG):] if idx != -1 else raw

        return answer.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mk_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Return a DataFrame even if *rows* is empty."""
    return pd.DataFrame(rows)


def _dump_answers(payload: List[Dict[str, str]]) -> str:
    ts = int(time.time())
    fname = os.path.join(OUTPUT_GAIA_DIR, f"agent_answers_{ts}.json")
    with open(fname, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"💾 Answers saved to {fname}")
    return fname


def _download_task_file(
    task_id: str,
    file_name: str,
    base_url: str = DEFAULT_API_URL,
    timeout: int = 30,
) -> Optional[str]:
    """
        Try to download /files/{task_id} → tmp/<file_name>.
        Returns '' on any failure instead of raising.
        """
    safe_name = os.path.basename(file_name) or f"{task_id}.bin"
    dest_path = os.path.join(QUESTIONS_FILES_DIR, safe_name)
    url = f"{base_url.rstrip('/')}/files/{task_id}"

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()                        # 4xx / 5xx → HTTPError
        with open(dest_path, "wb") as fh:              # disk I/O may fail
            fh.write(resp.content)
        return dest_path

    except (requests.exceptions.RequestException, OSError) as err:
        print(f"Could not fetch {url} -> {dest_path}: {err}")
        return None                                    # empty path signals failure

# ─────────────────────────────────────────────────────────────────────────────
# Core runner (streaming generator)
# ─────────────────────────────────────────────────────────────────────────────


def run_and_submit_all(
    profile: gr.OAuthProfile | None,
    stop_dict: dict,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
) -> Generator[tuple[str, pd.DataFrame, str | None], None, None]:
    """Stream Q&A, support early stop, submit at the end."""

    stop_dict["stop"] = False  # reset every run

    if profile is None:
        yield "🔒 Please log‑in with the HF button first.", _mk_df([]), None
        return

    username = profile.username
    yield f"### 👋 Welcome **{username}** – starting …", _mk_df([]), None

    # Build agent
    try:
        agent = BasicAgent()
    except Exception as exc:
        yield f"❌ Failed to initialise agent: {exc}", _mk_df([]), None
        return

    # Fetch questions
    q_url = f"{DEFAULT_API_URL}/questions"
    try:
        resp = requests.get(q_url, timeout=15)
        resp.raise_for_status()
        questions: List[Dict[str, Any]] = resp.json()
        if not questions:
            raise ValueError("question list is empty")
    except Exception as exc:
        yield f"❌ Could not fetch questions: {exc}", _mk_df([]), None
        return

    total_q = len(questions)
    yield f"### 📑 Fetched **{total_q}** questions.", _mk_df([]), None

    answers_payload: List[Dict[str, str]] = []
    qa_log: List[Dict[str, str]] = []
    results_log: List[Dict[str, str]] = []

    with tqdm(total=total_q, desc="Answering", unit="q", leave=False, disable=True) as bar:
        for idx, q in enumerate(questions, 1):
            if stop_dict.get("stop"):
                yield "🛑 Run cancelled by user (before finishing).", _mk_df(results_log), None
                return

            task_id, question_text, file = q.get("task_id"), q.get(
                "question"), q.get("file_name")
            if not task_id or question_text is None:
                answered = "⚠️ malformed question payload"
            else:
                try:
                    if file:
                        # Download the file if it exists
                        file_path = _download_task_file(task_id, file)
                        if not file_path:
                            file_path = None
                    else:
                        file_path = None
                    answered = agent(question=question_text,
                                     input_file=file_path)
                except Exception as exc:
                    answered = f"AGENT ERROR: {exc}"

            if task_id:
                answers_payload.append(
                    {"task_id": task_id, "submitted_answer": answered})
                qa_log.append({
                    "q_progress": f"{idx}/{total_q}",
                    "task_id": task_id,
                    "question": question_text,
                    "submitted_answer": answered,
                })

            results_log.append({
                "Q. Number": f"{idx}/{total_q}",
                "Question": question_text or "<missing>",
                "Submitted Answer": answered,
            })
            bar.update(1)
            yield f"### ✅ {idx}/{total_q} answered", _mk_df(results_log), None

    answers_file = _dump_answers(answers_payload)

    if stop_dict.get("stop"):
        yield "🛑 Run cancelled – answers saved locally, submission skipped.", _mk_df(results_log), answers_file
        return

    yield "### 💾 Answers saved – preparing submission …", _mk_df(results_log), answers_file

    if not answers_payload:
        yield "❌ No answers to submit.", _mk_df(results_log), answers_file
        return

    submit_url = f"{DEFAULT_API_URL}/submit"
    space_id = os.getenv("SPACE_ID", FALLBACK_SPACE_ID)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "<local-run>"

    submission = {"username": username,
                  "agent_code": agent_code, "answers": answers_payload}

    try:
        resp = requests.post(submit_url, json=submission, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        msg = (
            "## 🎉 Submission successful\n" +
            f"Score: **{data.get('score', 'N/A')}%** "
            f"({data.get('correct_count', '?')}/{data.get('total_attempted', '?')})\n" +
            f"Message: {data.get('message', '')}"
        )
        yield msg, _mk_df(results_log), answers_file
    except requests.exceptions.HTTPError as e:
        detail = f"Server responded with status {e.response.status_code}."
        try:
            err_json = e.response.json()
            detail += f" Detail: {err_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            detail += f" Response: {e.response.text[:500]}"
        yield f"❌ Submission failed: {detail}", _mk_df(results_log), answers_file
    except requests.exceptions.Timeout:
        yield "❌ Submission failed: request timed‑out.", _mk_df(results_log), answers_file
    except requests.exceptions.RequestException as e:
        yield f"❌ Submission failed: network error – {e}", _mk_df(results_log), answers_file
    except Exception as e:
        yield f"❌ Unexpected submission error: {e}", _mk_df(results_log), answers_file


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
#status_box {font-size:1.5rem; line-height:1.4; white-space:pre-line;}
#stop_button {background-color:#d9534f !important; color:white !important;}
#answers_table td {font-size:1rem;}
"""

demo = gr.Blocks(title="Agent Evaluation – Streaming Edition", css=CSS)

with demo:
    gr.Markdown("""# 🏃‍♂️ Live Agent Evaluation
    Watch answers stream in real‑time. Hit **Stop** to abort and download your
    work-in-progress JSON.
    """)

    gr.LoginButton()
    stop_state = gr.State({"stop": False})

    with gr.Row():
        run_btn = gr.Button(
            "Run Evaluation & Submit All Answers", variant="primary")
        stop_btn = gr.Button("Stop", elem_id="stop_button")

    status_box = gr.Markdown("Waiting …", elem_id="status_box")
    table = gr.DataFrame(elem_id="answers_table", interactive=False)
    dl_file = gr.File(label="Download answers JSON", interactive=False)

    # Start run; HF OAuth profile auto‑injected as 1st arg, we supply stop_state
    run_btn.click(
        run_and_submit_all,
        inputs=[stop_state],
        outputs=[status_box, table, dl_file],
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")  # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(
            f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:  # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(
            f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)
