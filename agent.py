# ==============================================================================
#
# LangGraph "Search-then-Read" RAG Agent (Wikipedia-Only Edition)
#
# This script implements a highly sophisticated conversational agent that mimics
# a human research process to answer questions using Wikipedia as its sole
# source of external information.
#
# Key Components:
# 1. Constrained Two-Step RAG ("Search-then-Read"): The agent has two tools
#    that guarantee a Wikipedia-only workflow:
#    a. `wikipedia_url_searcher`: A Tavily-based tool hard-coded to search
#       only within en.wikipedia.org to find relevant URLs.
#    b. `web_page_reader`: A tool that reads the full content of a URL provided
#       by the searcher.
# 2. Advanced Agentic Reasoning: The agent's core prompt instructs it on this
#    specific two-step research process, while also allowing it to answer
#    from its own knowledge if possible.
# 3. CriticNode: Acts as a final quality gate on the agent's comprehensive answer,
#    requesting revisions if the answer is unsatisfactory.
#
# ==============================================================================

# --- Core Imports ---
import os
import sys
import logging
from datetime import datetime
from typing import List, TypedDict

# Set a user agent to identify requests from our script. This is placed here to
# ensure the variable is set before any third-party libraries that might need it
# are imported.
os.environ["USER_AGENT"] = "LangGraph RAG Agent/1.0"

# --- Third-party Library Imports ---
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs): return text
    logging.warning("'termcolor' not found; install for colored output.")

# ==============================================================================
# SCRIPT SETUP AND CONFIGURATION
#
# This block handles all the preliminary setup required for the script to run.
# It configures logging, loads API keys from a secure file, sets up environment
# variables for services like LangSmith, defines the AI models to use, and
# performs essential checks to ensure all required configurations are in place
# before the main application logic begins.
# ==============================================================================
debug_mode = '--debug' in sys.argv
logging.basicConfig(filename='debug.log', level=logging.DEBUG if debug_mode else logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LangGraph Wikipedia Agent")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

AGENT_MODEL = "gemini-2.0-flash"
CRITIC_MODEL = "gemini-2.5-flash"

google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    raise RuntimeError("GOOGLE_API_KEY not set.")
if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError("TAVILY_API_KEY not set.")

# ==============================================================================
# TOOLS (Constrained to Wikipedia)
#
# This section defines the agent's capabilities. The agent is designed with a
# "Search-then-Read" architecture, which mimics a human research workflow. This
# requires two distinct tools: one to find relevant sources and another to read
# their content.
# ==============================================================================

# The base Tavily search engine instance, configured to fetch 5 results.
tavily_engine = TavilySearch(max_results=5)

# Tool 1: The "Searcher" - This tool is hard-coded to search only within Wikipedia.
@tool("wikipedia_url_searcher", description="A tool to search for Wikipedia pages and return a list of relevant URLs.")
def wikipedia_url_searcher(query: str) -> str:
    """
    Performs a site-specific web search on en.wikipedia.org using the Tavily engine.
    This gives us the powerful semantic search of Tavily, focused on the high-quality
    data source of Wikipedia.
    """
    # By appending "site:en.wikipedia.org", we instruct the search engine to
    # limit its results exclusively to pages from that domain.
    search_query = f"{query} site:en.wikipedia.org"
    return tavily_engine.invoke(search_query)

# Tool 2: The "Reader" - This tool reads the full content of a URL found by the searcher.
@tool("web_page_reader", description="A tool to read the full text content of a single web page URL.")
def web_page_reader(url: str) -> str:
    """
    Takes a URL, loads its content using WebBaseLoader, and returns the clean text.
    """
    try:
        # WebBaseLoader is a robust way to fetch and parse HTML content.
        loader = WebBaseLoader(url)
        docs = loader.load()
        # Join the content and truncate to a manageable size for the LLM context.
        full_content = "\n\n".join(doc.page_content for doc in docs)
        return full_content[:15000]
    except Exception as e:
        return f"Error reading URL: {e}"

# This is the final list of tools that will be made available to the agent's LLM.
tools = [wikipedia_url_searcher, web_page_reader]

# ==============================================================================
# GRAPH STATE, NODES, AND LOGIC
#
# This section defines the core components of our stateful agent using LangGraph.
# - The "State" is the memory of the system.
# - "Nodes" are the fundamental actors or functions (like the Agent or Critic).
# - "Logic" (or edges) defines the pathways and decisions that connect the nodes.
# ==============================================================================
class AgentState(TypedDict):
    """
    Defines the structure of the state object that is passed between nodes in
    the graph. It acts as the shared memory for the entire workflow.
    """
    messages: List[BaseMessage]
    critique: str
    iterations: int
    today: str
    step: int

class AgentNode:
    """The 'brain' of the operation. This node is responsible for reasoning,
    planning the research strategy (i.e., choosing which tool to use), and
    synthesizing the final answer from the gathered information."""
    def __init__(self, llm: ChatGoogleGenerativeAI, tools: list):
        # Bind the tools to the LLM. This allows the LLM to generate tool-calling
        # instructions in a structured format that LangGraph can understand.
        self.llm = llm.bind_tools(tools)

    def run(self, state: AgentState):
        """Executes the agent's logic for a single pass."""
        if debug_mode:
            # Print the main iteration header only on the first step of a loop.
            if state['step'] == 1:
                print(colored(f"\n--- AGENT ITERATION {state['iterations'] + 1} ---", 'magenta'))

            # Make the "Thinking..." step more descriptive based on the conversation history.
            step_description = "Synthesizing final answer..."
            last_message = state['messages'][-1]
            if isinstance(last_message, HumanMessage) or state.get('critique'):
                step_description = "Planning next action..."
            elif isinstance(last_message, ToolMessage):
                ai_msg = next((m for m in reversed(state['messages']) if isinstance(m, AIMessage) and m.tool_calls), None)
                if ai_msg and ai_msg.tool_calls[0]['name'] == 'wikipedia_url_searcher':
                    step_description = "Deciding which page to read..."
            print(colored(f"  Step {state['step']}: {step_description}", 'blue'))

        # The system prompt is the agent's core instruction set.
        agent_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI research assistant. Your only source of external information is Wikipedia.

YOUR TASK:
- First, determine if you can answer the user's query from your own knowledge. For common knowledge questions (e.g., "Who wrote Hamlet?"), answer directly.
- If you do not know the answer or the question requires current information, you MUST use your tools to research it.

YOUR RESEARCH WORKFLOW:
1.  Use the `wikipedia_url_searcher` tool to find the URLs of relevant Wikipedia pages.
2.  Analyze the search results and identify the single most promising URL.
3.  Use the `web_page_reader` tool with the best URL to get the full page content.
4.  Synthesize a comprehensive answer based on the full content you have read.

Today's Date: {today}
Previous Critique: {critique}"""),
            *state['messages']
        ])
        
        chain = agent_prompt | self.llm
        response = chain.invoke({"today": state['today'], "critique": state.get('critique', '')})
        return {"messages": state["messages"] + [response], "step": state["step"] + 1}

def tool_executor_node(state: AgentState):
    """
    The "hands" of the agent. This node is a dedicated, robust function for
    executing any tool calls the AgentNode decides to make.
    """
    if debug_mode:
        # The tool_call object is a dictionary, so we use key access ['name'].
        tool_name = state["messages"][-1].tool_calls[0]['name']
        print(colored(f"  Step {state['step']}: Executing tool '{tool_name}'...", "cyan"))

    tool_calls = state["messages"][-1].tool_calls
    tool_output_messages = []
    
    for call in tool_calls:
        if call["name"] == "web_page_reader":
            query = call["args"].get("url")
        else:
            query = call["args"].get("query")
        if query is None: query = next(iter(call["args"].values()), None)
        
        tool_to_call = next(t for t in tools if t.name == call["name"])
        output = tool_to_call.invoke(query)
        tool_output_messages.append(ToolMessage(content=str(output), tool_call_id=call['id']))
    
    # The tool execution step is complete, so we increment the step counter.
    return {"messages": state["messages"] + tool_output_messages, "step": state["step"] + 1}

class CriticNode:
    """The 'peer reviewer' of the operation. This node evaluates the agent's
    final answer for quality and accuracy."""
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_template(
"""You are a strict AI critic. Your job is to evaluate if the agent's last response successfully answers the original user query.
Today's Date: {today}
CRITICAL INSTRUCTIONS:
- If the agent's response is an admission of failure, you MUST respond with `REVISE`.
- If the final answer is accurate and complete, write `ACCEPT`.
- If the answer is incomplete or factually incorrect, write `REVISE` and provide a concise reason.
Conversation History:
{conversation_history}
Original User Query: {original_query}
Your Critique:"""
        )

    def run(self, state: AgentState):
        if debug_mode: print(colored(f"  Step {state['step']}: Evaluating final answer...", 'blue'))
        history_str = "\n".join([f"{type(msg).__name__}: {getattr(msg, 'content', str(msg))}" for msg in state["messages"]])
        chain = self.prompt_template | self.llm | StrOutputParser()
        critique = chain.invoke({
            "original_query": state['messages'][0].content, "conversation_history": history_str, "today": state['today']
        })
        if debug_mode: print(colored(f"Critic response: {critique}", 'yellow'))
        # The critic increments the main iteration counter and resets the step counter.
        return {"critique": critique, "iterations": state["iterations"] + 1, "step": 1}

def should_continue(state: AgentState):
    """
    The 'traffic cop' after the agent node. It directs the flow of the graph
    based on whether the agent decided to use a tool or give a final answer.
    """
    last_message = state["messages"][-1]
    if debug_mode:
        print(colored(f"  Router: Analyzing agent's last message (type: {type(last_message).__name__}).", "grey"))
    
    # Use hasattr for a safe check for the tool_calls attribute.
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        if debug_mode: print(colored("  Router: Decision -> Execute tools.", "grey"))
        return "execute_tools"
    
    if debug_mode: print(colored("  Router: Decision -> Go to critic.", "grey"))
    return "critic"

def after_critic_should_continue(state: AgentState) -> str:
    """
    The 'traffic cop' after the critic node. It decides whether to end the
    workflow or send the agent back for revisions.
    """
    if state["iterations"] > 3:
        return END
    if state.get("critique", "").strip().upper().startswith("REVISE"):
        return "agent"
    return END

# ==============================================================================
# # GRAPH COMPILATION AND MAIN EXECUTION
#
# This final section brings everything together. First, we define and build the
# state graph by connecting all our nodes and logic. Then, we compile it into a
# runnable application. The `main` function provides an interactive
# command-line interface for the user to chat with the compiled agent.
# ==============================================================================
# 1. Instantiate the nodes
agent_node = AgentNode(llm=ChatGoogleGenerativeAI(model=AGENT_MODEL, temperature=0, google_api_key=google_key), tools=tools)
critic_node = CriticNode(llm=ChatGoogleGenerativeAI(model=CRITIC_MODEL, temperature=0, google_api_key=google_key))

# 2. Build the state graph
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node.run)
graph_builder.add_node("tool_executor", tool_executor_node)
graph_builder.add_node("critic", critic_node.run)

# 3. Define the graph's edges and conditional routing
graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", should_continue, {"execute_tools": "tool_executor", "critic": "critic"})
graph_builder.add_edge("tool_executor", "agent") # After tools are run, always return to the agent.
graph_builder.add_conditional_edges("critic", after_critic_should_continue, {"agent": "agent", END: END})

# 4. Compile the graph into a runnable application object.
app = graph_builder.compile()

def main():
    """Sets up and runs the interactive command-line chat application."""
    print(colored("Welcome to the Wikipedia-powered AI Agent!", "cyan"))
    print("Type your question (or 'exit' to quit):")
    while True:
        try:
            q = input(colored("Your question: ", "green")).strip()
            if not q or q.lower() == "exit":
                print(colored("\nGoodbye!", "cyan"))
                sys.exit(0)
            
            today_str = datetime.now().strftime("%Y-%m-%d")
            
            # Create the initial state for the graph for a new query.
            initial_state: AgentState = {
                "messages": [HumanMessage(content=q)], "critique": "", "iterations": 0, "today": today_str, "step": 1,
            }
            
            # Invoke the graph with the initial state.
            final_state = app.invoke(initial_state)
            final_answer = final_state["messages"][-1]
            
            print(colored("\n--- Final Answer ---", "cyan"))
            print(colored(final_answer.content, "white"))
            print()
            
        except KeyboardInterrupt:
            print(colored("\nGoodbye!", "cyan"))
            sys.exit(0)
        except Exception as e:
            # Print the full traceback in debug mode for easier debugging.
            if debug_mode:
                import traceback
                traceback.print_exc()
            else:
                print(colored(f"\nAn unexpected error occurred: {e}", "red"))
            break

# Standard Python entry point.
if __name__ == "__main__":
    main()
