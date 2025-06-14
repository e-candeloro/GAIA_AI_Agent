import json
import os
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

import tiktoken  # pip install tiktoken
import trafilatura  # pip install trafilatura
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
# from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq
from langchain_huggingface import (ChatHuggingFace, HuggingFaceEmbeddings,
                                   HuggingFaceEndpoint)
from langchain_ollama.chat_models import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langmem.short_term import SummarizationNode  # NEW

from tools import get_tools

MAX_WINDOW_TOKENS = 3000          # absolute budget we send to Groq
RAW_LIMIT_BEFORE_SUM = 2000       # when raw history grows past this, summarise
SUMMARY_CHUNK = 800               # target length of the running summary
# import tools

# good fallback for Groq models
ENCODING = tiktoken.get_encoding("cl100k_base")


def n_tokens(txt: str) -> int:
    return len(ENCODING.encode(txt))


def tokens_in_messages(msgs: list) -> int:
    return sum(n_tokens(m.content) for m in msgs)


# SYSTEM_PROMPT = f"""
# You are a helpful assistant tasked with answering questions using a set of tools.
# You can use these tools to search for information, perform calculations, and retrieve data from various sources.
# If the tool is not available, you can try to find the information online. You can also use your own knowledge to answer the question.
# You need to provide a step-by-step explanation of how you arrived at the answer.

# You have access to the following tools:
# {tools}

# {file_info}
# ==========================
# Here is a few examples from humans, showing you how to answer the question step by step.

# Question 1: In terms of geographical distance between capital cities, which 2 countries are the furthest from each other within the ASEAN bloc according to wikipedia? Answer using a comma separated list, ordering the countries by alphabetical order.
# Steps:
# 1. Search the web for "ASEAN bloc".
# 2. Click the Wikipedia result for the ASEAN Free Trade Area.
# 3. Scroll down to find the list of member states.
# 4. Click into the Wikipedia pages for each member state, and note its capital.
# 5. Search the web for the distance between the first two capitals. The results give travel distance, not geographic distance, which might affect the answer.
# 6. Thinking it might be faster to judge the distance by looking at a map, search the web for "ASEAN bloc" and click into the images tab.
# 7. View a map of the member countries. Since they're clustered together in an arrangement that's not very linear, it's difficult to judge distances by eye.
# 8. Return to the Wikipedia page for each country. Click the GPS coordinates for each capital to get the coordinates in decimal notation.
# 9. Place all these coordinates into a spreadsheet.
# 10. Write formulas to calculate the distance between each capital.
# 11. Write formula to get the largest distance value in the spreadsheet.
# 12. Note which two capitals that value corresponds to: Jakarta and Naypyidaw.
# 13. Return to the Wikipedia pages to see which countries those respective capitals belong to: Indonesia, Myanmar.
# Tools:
# 1. Search engine
# 2. Web browser
# 3. Microsoft Excel / Google Sheets
# Final Answer: Indonesia, Myanmar

# Your Actions, to follow the human example, should be similar to the following:
# 1. Use the wiki_search tool to search for the ASEAN Free Trade Area.
# 2. Retrieve the list of member states from the Wikipedia page, and note their capitals if they are available. If not, use the web_search tool to find the capitals.
# 3. Once you have the capitals lists, use the web_search tool to find the GPS coordinates of each capital city.
# 3. Calculate the geographical distance between each pair of capitals. You can search for a formula to calculate the distance between two GPS coordinates, then use the calculator tool to perform the calculations.
# 4. Identify the pair of capitals with the maximum distance.
# 5. Provide the final answer in a comma-separated list, ordering the countries by alphabetical order.
# Final Answer: Indonesia, Myanmar
# ==========================

# IMPORTANT: if you are not able to answer the question, even with the help of the tools, you MUST say "I don't know" instead of making up an answer!!!
# ONLY finish your answer with the following template: [FINAL ANSWER]. The [FINAL ANSWER] should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.

# Example:

# Question: What is the capital of France?

# Output: [FINAL ANSWER] Paris

# I REPEAT: DON'T SAY ANYTHING ELSE, EVEN IF YOU THINK, THAN THE FINAL ANSWER in the format [FINAL ANSWER] *answer here*, and don't use any other format than the one specified above.
# """


def build_graph():
    """
    Build the state graph for the agent.
    This function defines the nodes and edges of the graph.
    """

    load_dotenv()

    BASE_PROMPT_PATH = Path(__file__).with_name("base_prompt.txt")
    BASE_PROMPT_TMPL = PromptTemplate(
        template=BASE_PROMPT_PATH.read_text(encoding="utf-8"),
        input_variables=["tools", "file_info"],   # placeholders we just added
    )
    TOOLS = get_tools()
    # setup llm and tools
    # llm = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash-preview-05-20", google_api_key=os.getenv("GOOGLE_API_KEY"))
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0,
                   api_key=os.getenv("GROQ_API_KEY"))

    llm_with_tools = llm.bind_tools(TOOLS)

    # define agent state
    class AgentState(TypedDict):
        """State for the agent."""
        # TODO: Add any additional state variables we need
        input_file: Optional[str]  # Contains file path (various formats)
        messages: Annotated[list[AnyMessage], add_messages]

    # define the assistant node
# ───────── updated assistant node ─────────

    def assistant(state: AgentState):
        file_path = state.get("input_file")
        tools_str = ", ".join(t.name for t in TOOLS)
        file_info = (
            f"You have also received an input file at `{file_path}`. "
            "Use an appropriate tool to read or analyse it."
            if file_path else
            "No input file has been provided for this question."
        )

        sys_msg = SystemMessage(
            content=BASE_PROMPT_TMPL.format(
                tools=tools_str, file_info=file_info)
        )
        full_msgs = [sys_msg] + state["messages"]

        # ── DEBUG 1: tokens of prompt messages ────────────────────────
        prompt_tok = tokens_in_messages(full_msgs)

        # ── DEBUG 2: tokens of tool function-schemas sent to Groq ─────
        fn_schemas = [convert_to_openai_function(t) for t in TOOLS]
        schema_tok = n_tokens(json.dumps(fn_schemas))

        print(f"[DEBUG] tokens → prompt={prompt_tok}  "
              f"tool-schema≈{schema_tok}  total≈{prompt_tok + schema_tok}")

        # ── call the model ────────────────────────────────────────────
        ai_msg = llm_with_tools.invoke(full_msgs)

        usage = ai_msg.additional_kwargs.get("usage_metadata", {})
        if usage:
            print(f"[DEBUG] Groq usage → in={usage.get('input_tokens')}  "
                  f"out={usage.get('output_tokens')}  "
                  f"total={usage.get('total_tokens')}")

        return {"messages": [ai_msg]}

    # ✨ 1)  Create a lightweight summarisation model (same base LLM)
    summarizer_model = llm.bind(max_tokens=128, temperature=0)

    # ✨ 2)  Configure SummarizationNode
    summarization_node = SummarizationNode(
        token_counter=count_tokens_approximately,
        model=summarizer_model,
        max_tokens=MAX_WINDOW_TOKENS,
        max_tokens_before_summary=RAW_LIMIT_BEFORE_SUM,
        max_summary_tokens=SUMMARY_CHUNK,
    )

    # ----- graph wiring --------------------------------------------------
    builder = StateGraph(AgentState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools=TOOLS))
    # builder.add_node("memory", summarization_node)     # NEW

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message requires a tool, route to tools
        # Otherwise, provide a direct response
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    react_graph = builder.compile()

    return react_graph


if __name__ == "__main__":
    # debug
    load_dotenv()

    class BasicAgent:
        """Shim around the langgraph returned by `build_graph()` that
        extracts only what follows the `[FINAL ANSWER]` tag."""

        TAG = "[FINAL ANSWER]"

        def __init__(self) -> None:
            print("⏳  Initialising BasicAgent …")
            self.graph = build_graph()
            print("✅  BasicAgent ready!")

        def __call__(self,
                     question: str,
                     input_file: Optional[str] = None) -> str:
            """Run the graph and return just the text after `[FINAL ANSWER]`."""
            msgs = [HumanMessage(content=question)]
            out = self.graph.invoke({"messages": msgs,
                                    "input_file": input_file})
            raw = out["messages"][-1].content

            idx = raw.rfind(self.TAG)
            return raw[idx + len(self.TAG):].strip() if idx != -1 else raw.strip()

    # test the agent
    agent = BasicAgent()
    print(agent(question="Describe this image: https://www.wikiwand.com/en/articles/Cat#/media/File:Cat_August_2010-4.jpg",
                input_file=None))
