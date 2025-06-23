import os
import json
import random
import re
import time
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WikipediaLoader, YoutubeLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables
load_dotenv()

# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two integers."""
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """Divide two integers."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def youtube_transcript(video_url: str) -> str:
    """
    Retrieve the transcript (captions) of a YouTube video, if available.
    This tool extracts and returns the full transcript text from the given YouTube video URL. It is helpful for answering questions based on what is said in a video, such as summarizing content or pulling out spoken facts. It does not interpret visual elements, only spoken audio with captions.
    """
    docs = YoutubeLoader.from_youtube_url(video_url).load()
    return "\\n".join([doc.page_content for doc in docs])

@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results.
    
    Args:
        query: The search query."""
    print(f"🔍 Tool 'wwiki_search' invoked with query: {query}")
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\\n\\n---\\n\\n".join(
        [f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\\n{doc.page_content}\\n</Document>' for doc in search_docs]
    )
    return {"wiki_results": formatted_search_docs}

@tool
def web_search(query: str) -> str:
    """Search Tavily Web for a query and return maximum 3 results from the web."""
    print(f"🔍 Tool 'web_search' invoked with query: {query}")
    search_tool = TavilySearchResults(max_results=3)
    search_results = search_tool.invoke(input=query)

    formatted = "\n\n---\n\n".join(
        f"<Document source=\"{r.get('source', '')}\"/>\n{r.get('content', '')}\n</Document>"
        for r in search_results
    )
    return formatted

class GeminiToolAgent:
    def __init__(self):
        system_prompt = """
        You are a helpful AI agent. After using tools or reasoning through a question, always return the answer on the last line in the format:
        FINAL ANSWER: <answer>

        You must reason step-by-step, use available tools when helpful, and produce a precise final answer.

        ==========================
        🔁 ANSWERING STRATEGY
        ==========================

        - Prioritize grounded evidence from tool outputs or the conversation context.
        - Do NOT speculate if evidence is missing.
        - Use web search if unable to answer using specific tools.

        ==========================
        🧠 REASONING FORMAT
        ==========================

        You must show your thought process, then conclude with this template:

        FINAL ANSWER: [A number OR short string OR comma-separated list of values]

        ==========================
        🚨 FINAL ANSWER RULES
        ==========================

        ❌ NEVER include explanation after FINAL ANSWER.
        ❌ NEVER include units (e.g., $, %, km) unless specifically requested.
        ❌ NEVER use commas in numbers (write 1000 instead of 1,000).
        ❌ NEVER use abbreviations or articles unless explicitly required.
        ❌ NEVER include icons or emojis before, within, or after the FINAL ANSWER

        Answer types:
        - Number → FINAL ANSWER: 42
        - String → FINAL ANSWER: Paris
        - Year → FINAL ANSWER: 2009
        - List → FINAL ANSWER: blue, green, red

        ⚠️ BEFORE you give your FINAL ANSWER:
        - Reread the question carefully.
        - Identify **exactly** what entity type is being asked (e.g., name, number, city, year).
        - Ensure the answer is **directly tied to the question**, not just something mentioned during reasoning.
        - Be careful to NOT give intermediate entities (e.g., actor name when the question asks for the character).

        ==========================
        ✅ FINAL CHECKLIST
        ==========================
        Before giving FINAL ANSWER:
        - [ ] Did I use all relevant tools and their results or grounded information?
        - [ ] Is my answer the specific format and type requested?
        - [ ] Did I avoid guessing or using unrelated intermediate facts?
        - [ ] Did I double check that my FINAL ANSWER matches the exact thing asked in the question?
        """

        self.sys_msg = SystemMessage(content=system_prompt)

        self.tools = [multiply, add, subtract, divide, web_search, wiki_search, youtube_transcript]
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        self.llm_with_tools = llm.bind_tools(self.tools)

        # Build LangGraph agent
        def assistant_node(state: MessagesState):
            result = self.llm_with_tools.invoke([self.sys_msg] + state["messages"])
            return {"messages": [result]}

        builder = StateGraph(MessagesState)
        builder.add_node("assistant_node", assistant_node)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "assistant_node")
        builder.add_conditional_edges("assistant_node", tools_condition)
        builder.add_edge("tools", "assistant_node")
        self.agent = builder.compile()

        self.BASE_BACKOFF = 30
        self.MAX_RETRIES = 5

    def extract_final_answer(self, submitted_text: str) -> str:
        match = re.search(r"FINAL ANSWER:\s*(.*)", submitted_text, re.IGNORECASE)
        return match.group(1).strip() if match else "N/A"

    def answer(self, question: str) -> str:
        attempt = 0
        while attempt < self.MAX_RETRIES:
            try:
                messages = self.agent.invoke({"messages": [HumanMessage(content=question)]}, debug=False)
                full_response = messages["messages"][-1].content
                print(f"Response: {full_response}")
                return self.extract_final_answer(full_response)
            except Exception as e:
                attempt += 1
                if "429" in str(e) or "rate limit" in str(e).lower():
                    if attempt < self.MAX_RETRIES:
                        wait_time = self.BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 1)
                        time.sleep(wait_time)
                    else:
                        return f"AGENT ERROR: Rate limit (after {self.MAX_RETRIES} attempts)"
                else:
                    return f"AGENT ERROR: {str(e)}"
        return "AGENT ERROR: Unknown"