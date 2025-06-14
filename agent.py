import os
from typing import Annotated, List, Optional, TypedDict

import trafilatura  # pip install trafilatura
from dotenv import load_dotenv
from langchain_core.documents import Document
# from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq
from langchain_huggingface import (ChatHuggingFace, HuggingFaceEmbeddings,
                                   HuggingFaceEndpoint)
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# import tools
from tools import get_tools


tools = get_tools()
SYSTEM_PROMPT = f"""
You are a helpful assistant tasked with answering questions using a set of tools.
You have access to the following tools:
{', '.join([tool.name for tool in tools])}
You can use these tools to search for information, perform calculations, and retrieve data from various sources.
If the tool is not available, you can try to find the information online. You can also use your own knowledge to answer the question. 
You need to provide a step-by-step explanation of how you arrived at the answer.

==========================
Here is a few examples from humans, showing you how to answer the question step by step.

Question 1: In terms of geographical distance between capital cities, which 2 countries are the furthest from each other within the ASEAN bloc according to wikipedia? Answer using a comma separated list, ordering the countries by alphabetical order.
Steps:
1. Search the web for "ASEAN bloc".
2. Click the Wikipedia result for the ASEAN Free Trade Area.
3. Scroll down to find the list of member states.
4. Click into the Wikipedia pages for each member state, and note its capital.
5. Search the web for the distance between the first two capitals. The results give travel distance, not geographic distance, which might affect the answer.
6. Thinking it might be faster to judge the distance by looking at a map, search the web for "ASEAN bloc" and click into the images tab.
7. View a map of the member countries. Since they're clustered together in an arrangement that's not very linear, it's difficult to judge distances by eye.
8. Return to the Wikipedia page for each country. Click the GPS coordinates for each capital to get the coordinates in decimal notation.
9. Place all these coordinates into a spreadsheet.
10. Write formulas to calculate the distance between each capital.
11. Write formula to get the largest distance value in the spreadsheet.
12. Note which two capitals that value corresponds to: Jakarta and Naypyidaw.
13. Return to the Wikipedia pages to see which countries those respective capitals belong to: Indonesia, Myanmar.
Tools:
1. Search engine
2. Web browser
3. Microsoft Excel / Google Sheets
Final Answer: Indonesia, Myanmar

Your Actions, to follow the human example, should be similar to the following:
1. Use the wiki_search tool to search for the ASEAN Free Trade Area.
2. Retrieve the list of member states from the Wikipedia page, and note their capitals if they are available. If not, use the web_search tool to find the capitals.
3. Once you have the capitals lists, use the web_search tool to find the GPS coordinates of each capital city.
3. Calculate the geographical distance between each pair of capitals. You can search for a formula to calculate the distance between two GPS coordinates, then use the calculator tool to perform the calculations.
4. Identify the pair of capitals with the maximum distance.
5. Provide the final answer in a comma-separated list, ordering the countries by alphabetical order.
Final Answer: Indonesia, Myanmar
==========================

IMPORTANT: if you are not able to answer the question, even with the help of the tools, you MUST say "I don't know" instead of making up an answer!!!
ONLY finish your answer with the following template: [FINAL ANSWER]. The [FINAL ANSWER] should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.

Example:

Question: What is the capital of France?

Output: [FINAL ANSWER] Paris
 
I REPEAT: DON'T SAY ANYTHING ELSE THAN THE FINAL ANSWER in the format [FINAL ANSWER] *answer here*, and don't use any other format than the one specified above.
"""


def build_graph():
    """
    Build the state graph for the agent.
    This function defines the nodes and edges of the graph.
    """

    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # setup llm and tools
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-05-20", google_api_key=GOOGLE_API_KEY)
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)

    # define agent state
    class AgentState(TypedDict):
        """State for the agent."""
        # TODO: Add any additional state variables we need
        messages: Annotated[list[AnyMessage], add_messages]

    # define the assistant node
    def assistant(state: AgentState):
        # System message
        sys_msg = SystemMessage(
            content=SYSTEM_PROMPT)

        return {
            "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        }

    # The graph
    builder = StateGraph(AgentState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message requires a tool, route to tools
        # Otherwise, provide a direct response
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    return builder.compile()


if __name__ == "__main__":
    # debug
    load_dotenv()

    class BasicAgent:
        """A langgraph agent."""

        def __init__(self):
            print("BasicAgent initialized.")
            self.graph = build_graph()

        def __call__(self, question: str) -> str:
            print(
                f"Agent received question (first 50 chars): {question[:50]}...")
            # Wrap the question in a HumanMessage from langchain_core
            messages = [HumanMessage(content=question)]
            messages = self.graph.invoke({"messages": messages})
            answer = messages['messages'][-1].content

            # If the answer does not contain [ANSWER], check if it starts with [ANSWER]
            if answer.startswith("[FINAL ANSWER]"):
                return answer[14:].strip()
            else:
                # If the answer does not start with [ANSWER], return the answer as is
                # This is a fallback in case the agent does not follow the template
                print(
                    "Warning: Answer does not start with [ANSWER]. Returning the answer as is.")
                return answer.strip()

    # test the agent
    agent = BasicAgent()
    print(agent("What is the capital of Italy?"))
