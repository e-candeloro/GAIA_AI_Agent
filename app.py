import os
import time
import json
from typing import Generator, List, Dict, Any

import gradio as gr
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm
from langchain_core.messages import HumanMessage

from agent import build_graph

load_dotenv()

# --- Constants ---------------------------------------------------------------
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
# fallback space ID for local runs, CHANGE THIS TO YOUR OWN SPACE ID IF NEEDED
FALLBACK_SPACE_ID = "ecandeloro/hf_agent_gaia_30"


# -----------------------------------------------------------------------------
# Agent wrapper ----------------------------------------------------------------
# -----------------------------------------------------------------------------


class BasicAgent:
    """Thin convenience wrapper around the langgraph returned by build_graph."""

    def __init__(self) -> None:
        print("⏳  Initialising BasicAgent …")
        self.graph = build_graph()
        print("✅  BasicAgent ready!")

    def __call__(self, question: str) -> str:
        print(f"⚙️   Processing: {question[:60]}…")
        messages = [HumanMessage(content=question)]
        chain_out = self.graph.invoke({"messages": messages})
        answer: str = chain_out["messages"][-1].content

        prefix = "[FINAL ANSWER]"
        if answer.startswith(prefix):
            answer = answer[len(prefix):]
        return answer.strip()


# -----------------------------------------------------------------------------
# Core runner ------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _mk_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Helper that always returns a DataFrame (Gradio dislikes `None`)."""
    return pd.DataFrame(rows, columns=["Task ID", "Question", "Submitted Answer"])


def _dump_answers(answers_payload: List[Dict[str, str]]) -> str:
    """Persist answers to a timestamped JSON file and return the path."""
    ts = int(time.time())
    fname = f"agent_answers_{ts}.json"
    with open(fname, "w", encoding="utf-8") as fh:
        json.dump(answers_payload, fh, ensure_ascii=False, indent=2)
    return fname


def run_and_submit_all(
    profile: gr.OAuthProfile | None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> Generator[tuple[str, pd.DataFrame, str | None], None, None]:
    """Main orchestration.

    Generates `(status_markdown, results_df, download_path)` tuples so the UI
    can update live. `download_path` is `None` until the question‑answering loop
    finishes, at which point it points to a JSON file containing all answers ‑
    handy if submission fails.
    """

    # 0️⃣  Early exit if not logged‑in
    if profile is None:
        yield "🔒 Please log‑in with the HF button first.", _mk_df([]), None
        return

    username = profile.username
    yield f"👋 Welcome **{username}** – starting …", _mk_df([]), None

    # 1️⃣  Instantiate agent ----------------------------------------------------
    try:
        agent = BasicAgent()
    except Exception as exc:
        yield f"❌ Failed to initialise agent: {exc}", _mk_df([]), None
        return

    # 2️⃣  Fetch questions ------------------------------------------------------
    questions_url = f"{DEFAULT_API_URL}/questions"
    try:
        resp = requests.get(questions_url, timeout=15)
        resp.raise_for_status()
        questions: List[Dict[str, Any]] = resp.json()
        if not questions:
            raise ValueError("question list is empty")
    except Exception as exc:
        yield f"❌ Could not fetch questions: {exc}", _mk_df([]), None
        return

    total_q = len(questions)
    yield f"📑 Fetched **{total_q}** questions.", _mk_df([]), None

    # 3️⃣  Iterate questions ----------------------------------------------------
    answers_payload: List[Dict[str, str]] = []
    results_log: List[Dict[str, str]] = []

    with tqdm(total=total_q, desc="Answering", unit="q", leave=False) as bar:
        for idx, q in enumerate(questions, 1):
            task_id, question_text = q.get("task_id"), q.get("question")
            if not task_id or question_text is None:
                answered = "⚠️ malformed question payload"
            else:
                try:
                    answered = agent(question_text)
                except Exception as exc:
                    answered = f"AGENT ERROR: {exc}"

            if task_id:
                answers_payload.append(
                    {"task_id": task_id, "submitted_answer": answered})

            results_log.append(
                {"Task ID": task_id or "?", "Question": question_text or "<missing>", "Submitted Answer": answered})
            bar.update(1)
            yield f"✅ Answered {idx}/{total_q}", _mk_df(results_log), None

    # 4️⃣  Persist answers so they are never lost ------------------------------
    answers_file = _dump_answers(answers_payload)
    yield "💾 Answers saved locally – preparing submission …", _mk_df(results_log), answers_file

    # 5️⃣  Submit ----------------------------------------------------------------
    if not answers_payload:
        yield "❌ No answers to submit.", _mk_df(results_log), answers_file
        return

    submit_url = f"{DEFAULT_API_URL}/submit"
    # fallback when running locally
    # fallback to a local run for this space. Change this to your own space ID!!!
    space_id = os.getenv("SPACE_ID", FALLBACK_SPACE_ID)
    agent_code_url = f"https://huggingface.co/spaces/{space_id}/tree/main" if space_id else "<local-run>"

    submission = {"username": username,
                  "agent_code": agent_code_url, "answers": answers_payload}

    try:
        resp = requests.post(submit_url, json=submission, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        status_msg = (
            "### 🎉 Submission successful\n" +
            f"Score: **{data.get('score', 'N/A')}%** "
            f"({data.get('correct_count', '?')}/{data.get('total_attempted', '?')})\n" +
            f"Message: {data.get('message', '')}"
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
        yield f"❌ Unexpected submission error: {e}", _mk_df(results_log), answers_file


# -----------------------------------------------------------------------------
# Gradio UI --------------------------------------------------------------------
# -----------------------------------------------------------------------------

demo = gr.Blocks(title="Agent Evaluation Runner – Streaming Edition")
with demo:
    gr.Markdown("""# 🏃‍♂️ Live Agent Evaluation
    Answers are streamed one‑by‑one. A JSON copy of all answers is always
    available for download so nothing is lost if submission fails.
    """)

    gr.LoginButton()

    run_btn = gr.Button(
        "Run Evaluation & Submit All Answers", variant="primary")

    status_box = gr.Markdown("Waiting …")
    table = gr.DataFrame(
        headers=["Task ID", "Question", "Submitted Answer"], interactive=False)
    dl_file = gr.File(label="Download answers JSON", interactive=False)

    run_btn.click(
        run_and_submit_all,
        inputs=[gr.OAuthProfile()],  # auto‑filled by LoginButton
        outputs=[status_box, table, dl_file],
    )

if __name__ == "__main__":
    demo.launch()
