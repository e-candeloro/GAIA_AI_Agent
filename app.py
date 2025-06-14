import json
import os
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent import build_graph

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
SUBMIT_ENDPOINT = f"{DEFAULT_API_URL.rstrip('/')}/submit"  # API endpoint

# If you fork this space, point this to *your* repo so the grader can pull code
FALLBACK_SPACE_ID = "ecandeloro/hf_agent_gaia_30"  # change when forking!

TEMP_DIR = os.getenv("TEMP_DIR", "./tmp")
QUESTIONS_FILES_DIR = os.path.join(TEMP_DIR, "questions_files")
OUTPUT_GAIA_DIR = os.path.join(TEMP_DIR, "output_gaia")

os.makedirs(QUESTIONS_FILES_DIR, exist_ok=True)
os.makedirs(OUTPUT_GAIA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Agent wrapper
# ─────────────────────────────────────────────────────────────────────────────


class BasicAgent:
    """Thin convenience wrapper around the langgraph returned by build_graph."""

    TAG = "[FINAL ANSWER]"

    def __init__(self) -> None:
        print("⏳  Initialising BasicAgent …")
        self.graph = build_graph()
        print("✅  BasicAgent ready!")

    def __call__(self, question: str, input_file: Optional[str] = None) -> str:
        msgs = [HumanMessage(content=question)]
        out = self.graph.invoke({"messages": msgs, "input_file": input_file})
        raw = out["messages"][-1].content
        idx = raw.rfind(self.TAG)
        return raw[idx + len(self.TAG):].strip() if idx != -1 else raw.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _mk_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Utility: consistently shaped DataFrame for the Gradio table."""
    return pd.DataFrame(rows, columns=["Task ID", "Question", "Submitted Answer"])


def _dump_answers(payload: List[Dict[str, str]]) -> str:
    ts = int(time.time())
    fname = os.path.join(OUTPUT_GAIA_DIR, f"agent_answers_{ts}.json")
    with open(fname, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"💾 Answers saved to {fname}")
    return fname


def _download_task_file(task_id: str, file_name: str, base_url: str = DEFAULT_API_URL, timeout: int = 30) -> Optional[str]:
    """Fetch an attachment for a question, streaming to disk."""
    if not file_name:
        return None
    safe_name = os.path.basename(file_name)
    dest_path = os.path.join(QUESTIONS_FILES_DIR, f"{task_id}_{safe_name}")
    url = f"{base_url.rstrip('/')}/files/{task_id}"
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with open(dest_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
        print(f"✅ Downloaded {url} → {dest_path}")
        return dest_path
    except (requests.exceptions.RequestException, OSError) as err:
        print(f"❌ Could not fetch {url}: {err}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────


def run_and_submit_all(
    profile: gr.OAuthProfile | None,
    stop_dict: dict,
) -> Generator[Tuple[str, pd.DataFrame, Optional[str]], None, None]:
    """Main coroutine executed by the UI.

    Streams markdown status, a DataFrame log, and an optional path to the saved‑answers JSON.
    The tqdm bar was removed because it does not surface in the Gradio frontend.
    """

    stop_dict["stop"] = False

    # 0️⃣  Auth check ---------------------------------------------------------
    if profile is None:
        yield "🔒 Please log‑in with the HF button first.", _mk_df([]), None
        return

    username = profile.username or "anonymous"
    yield f"### 👋 Welcome **{username}** – starting …", _mk_df([]), None

    # 1️⃣  Build agent --------------------------------------------------------
    try:
        agent = BasicAgent()
    except Exception as exc:
        yield f"❌ Failed to initialise agent: {exc}", _mk_df([]), None
        return

    # 2️⃣  Fetch questions ----------------------------------------------------
    try:
        resp = requests.get(
            f"{DEFAULT_API_URL.rstrip('/')}/questions", timeout=15)
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
    results_log: List[Dict[str, str]] = []

    # No tqdm – simple loop --------------------------------------------------
    for idx, q in enumerate(questions, 1):
        if stop_dict.get("stop"):
            yield "🛑 Run cancelled by user (before finishing).", _mk_df(results_log), None
            return

        task_id, question_text, file_name = q.get(
            "task_id"), q.get("question"), q.get("file_name")
        if not task_id or question_text is None:
            answered = "⚠️ malformed question payload"
        else:
            try:
                file_path = _download_task_file(task_id, file_name)
                answered = agent(question_text, input_file=file_path)
            except Exception as exc:
                answered = f"AGENT ERROR: {exc}"

        answers_payload.append(
            {"task_id": task_id, "submitted_answer": answered})
        results_log.append(
            {"Task ID": task_id, "Question": question_text, "Submitted Answer": answered})
        yield f"### ✅ {idx}/{total_q} answered", _mk_df(results_log), None

    # 3️⃣  Save locally -------------------------------------------------------
    answers_file = _dump_answers(results_log)
    yield "### 📥 Answers saved locally.", _mk_df(results_log), answers_file

    if stop_dict.get("stop"):
        yield "🛑 Run cancelled – answers saved locally, submission skipped.", _mk_df(results_log), answers_file
        return

    yield "### 📤 Submitting answers …", _mk_df(results_log), answers_file

    # 4️⃣  Build submission payload per spec ---------------------------------
    space_id = os.getenv("SPACE_ID", FALLBACK_SPACE_ID)
    agent_code_url = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "<local-run>"

    submission = {
        "username": username,
        "agent_code": agent_code_url,
        "answers": answers_payload,
    }

    # 5️⃣  POST to grading API -----------------------------------------------
    try:
        resp = requests.post(SUBMIT_ENDPOINT, json=submission, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        status_msg = (
            "### 🎉 Submission successful\n" +
            f"Score: **{data.get('score', 'N/A')}%** "
            f"({data.get('correct_count', '?')}/{data.get('total_attempted', '?')})\n\n" +
            f"{data.get('message', '')}"
        )
        yield status_msg, _mk_df(results_log), answers_file
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
        yield f"❌ Unexpected submission error: {e}", _mk_df(results_log), answers


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
    Answers are streamed one‑by‑one. A JSON copy of all answers is always available
    for download so nothing is lost if submission fails.
    """)

    gr.LoginButton()
    stop_state = gr.State({"stop": False})

    with gr.Row():
        run_btn = gr.Button(
            "Run Evaluation & Submit All Answers", variant="primary")
        stop_btn = gr.Button("Stop", elem_id="stop_button")

    status_box = gr.Markdown("Waiting …", elem_id="status_box")
    table = gr.DataFrame(headers=[
                         "Task ID", "Question", "Submitted Answer"], interactive=False, elem_id="answers_table")
    dl_file = gr.File(label="Download answers JSON", interactive=False)

    # Event listener for the long-running generator – returns a Dependency obj
    run_event = run_btn.click(
        run_and_submit_all,
        inputs=[stop_state],  # OAuth profile injected automatically
        outputs=[status_box, table, dl_file],
    )

    def _set_stop_flag(state):
        state["stop"] = True
        return state

    stop_btn.click(
        _set_stop_flag,
        inputs=[stop_state],
        outputs=[stop_state],
        cancels=[run_event],
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    print("Launching Gradio Interface for Basic Agent Evaluation…")
    # Enable queuing globally so the progress bar and async events work
    demo.queue()
    demo.launch(debug=True, share=False)
