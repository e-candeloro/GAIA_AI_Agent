import json
import os
import shutil
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from agent import build_graph, build_graph_with_react

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


# ─────────────────────────────────────────────────────────────────────────────
# Agent wrapper
# ─────────────────────────────────────────────────────────────────────────────


class GAIAAgent:
    """Thin convenience wrapper around the langgraph returned by build_graph."""

    TAG = "[FINAL ANSWER]"

    def __init__(self, model: str = "qwen/qwen3-32b", use_react_agent: bool = False) -> None:
        print("⏳  Initialising GAIA Agent …")
        self.model = model
        print(f"Using model: {self.model}")

        # check for required environment variables

        if not os.getenv("GROQ_API_KEY"):
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please set it to your Groq API key."
            )

        if not os.getenv("TAVILY_API_KEY"):
            raise ValueError(
                "TAVILY_API_KEY environment variable is not set. "
                "Please set it to your Tavily API key."
            )
        self.graph = build_graph() if not use_react_agent else build_graph_with_react()
        print("✅  GAIA Agent ready!")

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


def _download_task_file(
    task_id: str, file_name: str, base_url: str = DEFAULT_API_URL, timeout: int = 30
) -> Optional[str]:
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


def _reset_temp_dirs() -> None:
    """Ensure temp directories exist and are clean."""
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            print(
                f"🗑️ Removed temporary directory and its contents: {TEMP_DIR}")
    except Exception as e:
        print(f"❌ Failed to clean TEMP_DIR: {e}")

    for subdir in [TEMP_DIR, QUESTIONS_FILES_DIR, OUTPUT_GAIA_DIR]:
        try:
            os.makedirs(subdir, exist_ok=True)
        except Exception as e:
            print(f"❌ Could not create {subdir}: {e}")
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

    # Cleanup temporary files from previous runs ----------------------------------------
    _reset_temp_dirs()

    # 0️⃣  Auth check ---------------------------------------------------------
    if profile is None:
        yield "### 🔒 Please log‑in with the HF button first.", _mk_df([]), None
        return

    username = profile.username or "anonymous"
    yield f"### 👋 Welcome **{username}** – starting …", _mk_df([]), None

    # 1️⃣  Build agent --------------------------------------------------------
    try:
        # Fallback to a default model if not set as env var
        agent = GAIAAgent(model=os.getenv(
            "MODEL", "qwen/qwen3-32b"), use_react_agent=False)

        yield "### ✅ Agent initialised successfully.", _mk_df([]), None
    except Exception as exc:
        yield f"### ❌ Failed to initialise agent: {exc}", _mk_df([]), None
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
            yield (
                "### 🛑 Run cancelled by user (before finishing).",
                _mk_df(results_log),
                None,
            )
            return

        task_id, question_text, file_name = (
            q.get("task_id"),
            q.get("question"),
            q.get("file_name"),
        )
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
            {
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": answered,
            }
        )
        yield f"### ✅ {idx}/{total_q} answered", _mk_df(results_log), None

    # 3️⃣  Save locally -------------------------------------------------------
    answers_file = _dump_answers(results_log)
    yield "### 📥 Answers saved locally.", _mk_df(results_log), answers_file

    if stop_dict.get("stop"):
        yield (
            "### 🛑 Run cancelled – answers saved locally, submission skipped.",
            _mk_df(results_log),
            answers_file,
        )
        return

    yield "### 📤 Submitting answers …", _mk_df(results_log), answers_file

    # 4️⃣  Build submission payload per spec ---------------------------------
    space_id = os.getenv("SPACE_ID", FALLBACK_SPACE_ID)
    agent_code_url = (
        f"https://huggingface.co/spaces/{space_id}/tree/main"
        if space_id
        else "<local-run>"
    )

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
            "# 🎉 Submission successful\n"
            + f"## Score: **{data.get('score', 'N/A')}%** "
            f"({data.get('correct_count', '?')}/{data.get('total_attempted', '?')})\n\n"
            + f"## {data.get('message', '')}"
        )
        yield status_msg, _mk_df(results_log), answers_file
    except requests.exceptions.HTTPError as e:
        detail = f"Server responded with status {e.response.status_code}."
        try:
            err_json = e.response.json()
            detail += f" Detail: {err_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            detail += f" Response: {e.response.text[:500]}"
        yield f"### ❌ Submission failed: {detail}", _mk_df(results_log), answers_file
    except requests.exceptions.Timeout:
        yield (
            "### ❌ Submission failed: request timed‑out.",
            _mk_df(results_log),
            answers_file,
        )
    except requests.exceptions.RequestException as e:
        yield (
            f"### ❌ Submission failed: network error – {e}",
            _mk_df(results_log),
            answers_file,
        )
    except Exception as e:
        yield f"### ❌ Unexpected submission error: {e}", _mk_df(results_log)


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
    gr.Markdown("""
    # 🏃‍♂️ Live Agent Evaluation
    Answers are streamed one‑by‑one. A JSON copy of all answers is always available
    for download so nothing is lost if submission fails.

    ## How to use this app
    1. Log in with your Hugging Face account (top right corner).
    2. Click the **Run Evaluation & Submit All Answers** button.
    3. Wait for the evaluation to complete. You can stop it at any time by clicking the **Stop** button.
    4. Download the answers JSON file to keep a copy of your answers.
    5. If the submission is successful, you will see your score and a message from the grader.
    6. If the submission fails, you will see an error message and can still download the answers JSON file.

    ## Installation on Hugging Face Spaces or Locally

    1. Clone this space to your own account
    2. Set the `GROQ_API_KEY` and the `TAVILY_API_KEY` environment variables. You need to sign up in Groq and Tavily to get their API keys. They also have free tiers.
    3. If you want to run locally, set the `SPACE_ID` or the `FALLBACK_SPACE_ID` environment variable to your own space ID (e.g. `ecandeloro/hf_agent_gaia_30`).
    """)

    gr.LoginButton()
    stop_state = gr.State({"stop": False})

    with gr.Row():
        run_btn = gr.Button(
            "Run Evaluation & Submit All Answers", variant="primary")
        stop_btn = gr.Button("Stop", elem_id="stop_button")

    status_box = gr.Markdown("Waiting …", elem_id="status_box")
    table = gr.DataFrame(
        headers=["Task ID", "Question", "Submitted Answer"],
        interactive=False,
        elem_id="answers_table",
    )
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
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    print("Launching Gradio Interface for Basic Agent Evaluation…")
    # Enable queuing globally so the progress bar and async events work
    demo.queue()
    demo.launch(debug=True, share=False)
