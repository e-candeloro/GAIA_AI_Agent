from langgraph.prebuilt import create_react_agent
import os
from pathlib import Path
from typing import Annotated, Optional, TypedDict, List

import tiktoken
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools import get_tools


# ---------------------- helpers ------------------------------------------------
ENCODING = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


# ---------------------- graph builder -----------------------------------------
def build_graph(model: str = "qwen/qwen3-32b") -> StateGraph:
    load_dotenv()

    base_prompt = PromptTemplate(
        template=Path(__file__).with_name(
            "base_prompt.txt").read_text("utf-8"),
        input_variables=["tools", "file_info"],
    )

    TOOLS = get_tools()

    llm = ChatGroq(
        # valid Groq model id - less powerful but cheaper alternative: llama3-8b-8192
        model=model,
        temperature=0,  # for deterministic output
        api_key=os.getenv("GROQ_API_KEY"),
    )

    llm_with_tools = llm.bind_tools(TOOLS)

    # --- Graph state ----------------------------------------------------------
    class AgentState(TypedDict):
        input_file: Optional[str]
        messages: Annotated[List[AnyMessage], add_messages]

    # --- assistant node -------------------------------------------------------
    def assistant(state: AgentState):
        tools_str = ", ".join(t.name for t in TOOLS)
        file_info = (
            f"You have also received an input file at `{state['input_file']}`. "
            "Use an appropriate tool to read or analyse it."
            if state.get("input_file")
            else "No input file has been provided for this question."
        )

        sys_msg = SystemMessage(
            content=base_prompt.format(tools=tools_str, file_info=file_info)
        )

        ai_msg = llm_with_tools.invoke([sys_msg] + state["messages"])

        usage = ai_msg.additional_kwargs.get("usage_metadata", {})
        if usage:
            print(
                f"[Groq] in={usage.get('input_tokens')}  "
                f"out={usage.get('output_tokens')}  "
                f"total={usage.get('total_tokens')}"
            )
        return {"messages": [ai_msg]}

    # --- wire graph -----------------------------------------------------------
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools=TOOLS))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()


def build_graph_with_react(model: str = "qwen/qwen3-32b") -> StateGraph:
    load_dotenv()
    TOOLS = get_tools()

    # your custom prompt (still loaded from file)
    base_prompt = PromptTemplate(
        template=Path(__file__).with_name(
            "base_prompt.txt").read_text("utf-8"),
        input_variables=[],            # ReAct doesn’t substitute these automatically
    )

    llm = ChatGroq(model=model, temperature=0,
                   api_key=os.getenv("GROQ_API_KEY"))

    # Note: system_prompt expects a str, so use .template
    graph = create_react_agent(
        llm,
        tools=TOOLS,
        prompt=base_prompt.template,
    )
    return graph


# ---------------------- CLI test ----------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    graph = build_graph()
    out = graph.invoke(
        {
            "messages": [HumanMessage(content="What is the capital of France? ")],
            "input_file": "",
        }
    )
    print(out["messages"][-1].content)
