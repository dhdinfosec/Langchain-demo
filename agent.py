# ==============================================================================
#
# LangGraph Agent with a Critic
#
# This script implements a conversational agent using the LangGraph library.
# The agent is designed to answer questions, using a Wikipedia search tool for
# information beyond its knowledge cutoff. A second AI, the "critic,"
# evaluates the agent's responses for accuracy and completeness, creating a
# loop of feedback and refinement until the answer is satisfactory.
#
# ==============================================================================

# --- Core Imports ---
import os
import sys
import json
import re
import logging
from datetime import datetime
from typing import List, TypedDict, Optional

# --- Third-party Library Imports ---
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs): return text
    logging.warning("'termcolor' not found; install for colored output.")


# ==============================================================================
#
# SCRIPT SETUP AND CONFIGURATION
#
# ==============================================================================

debug_mode = '--debug' in sys.argv
if debug_mode:
    print(colored("DEBUG MODE ENABLED: verbose logging active.", 'yellow'))
    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LangGraph Agent with Critic")

AGENT_MODEL = "gemini-2.0-flash"
CRITIC_MODEL = "gemini-2.5-flash"

google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    raise RuntimeError("GOOGLE_API_KEY not set in environment or .env file.")


# ==============================================================================
#
# MODELS AND TOOLS INITIALIZATION
#
# ==============================================================================

agent_llm = ChatGoogleGenerativeAI(model=AGENT_MODEL, temperature=0, google_api_key=google_key)
critic_llm = ChatGoogleGenerativeAI(model=CRITIC_MODEL, temperature=0, google_api_key=google_key)

@tool("wikipedia_search", description="Search Wikipedia and return a summary.")
def wikipedia_search(query: str) -> str:
    """A tool that allows the agent to search Wikipedia."""
    try:
        import wikipedia
        if debug_mode:
            print(colored(f"DEBUG: Wikipedia search for '{query}'", "yellow"))
        return wikipedia.summary(query, sentences=5, auto_suggest=False)
    except wikipedia.exceptions.PageError:
        if debug_mode:
            print(colored(f"DEBUG: PageError, retry with auto_suggest for '{query}'", "yellow"))
        try:
            return wikipedia.summary(query, sentences=5, auto_suggest=True)
        except Exception as e:
            return f"Wikipedia fallback error: {e}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Wikipedia disambiguation error for query '{query}': Options: {e.options[:5]}"
    except Exception as e:
        return f"Wikipedia search error: {e}"

tools = [wikipedia_search]


# ==============================================================================
#
# GRAPH STATE DEFINITION
#
# ==============================================================================

class AgentState(TypedDict):
    """Represents the state of the agent workflow."""
    messages: List[BaseMessage]
    critique: str
    iterations: int


# ==============================================================================
#
# AGENT NODE
#
# ==============================================================================

class AgentNode:
    """The node responsible for executing the main agent logic."""
    def __init__(self, llm: ChatGoogleGenerativeAI, tools: List[tool]):
        self.llm = llm.bind_tools(tools)
        self.tools = {t.name: t for t in tools}
        today = datetime.now().strftime("%Y-%m-%d")

        system_message = f"""You are a helpful AI assistant with access to a wikipedia_search tool. Your goal is to answer the user's query accurately and concisely.

Your knowledge cutoff is June 2024. Today's date is {today}.

Instructions:
1.  Examine the user's query.
2.  If the query is about information or events after your knowledge cutoff, you must use the `wikipedia_search` tool.
3.  If you receive a critique that your answer was incomplete or incorrect, you MUST try a new tool call with a different, more specific search query to find the correct information.
4.  If after a few attempts the tool is not providing a useful answer, inform the user that you were unable to find a definitive answer.
5.  Once you have a satisfactory answer from the tool, synthesize it into a final, user-facing response. DO NOT output the raw tool content.

Previous Critique:
{{critique}}"""

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{user_query}")
        ])

    def run(self, state: AgentState):
        """The main execution method for the agent node."""
        if debug_mode:
            print(colored(f"\nAGENT NODE ITERATION {state['iterations'] + 1}", 'magenta'))
            if state.get("critique"):
                print(colored(f"Critique: {state['critique']}", 'yellow'))

        chain = self.prompt_template | self.llm
        response = chain.invoke({
            "critique": state.get("critique", ""),
            "user_query": state["messages"][0].content,
            "messages": state["messages"]
        })

        if not response.tool_calls:
            if debug_mode:
                print(colored("Agent provides direct answer.", 'cyan'))
            return {"messages": state["messages"] + [response]}

        if debug_mode:
            print(colored("Agent requested tool call.", 'cyan'))
        
        tool_output_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            if debug_mode:
                print(colored(f"Calling tool '{tool_name}' with input {tool_input}", 'cyan'))

            try:
                output = self.tools[tool_name].invoke(tool_input)
            except TypeError:
                output = self.tools[tool_name].invoke(next(iter(tool_input.values())))

            if debug_mode:
                print(colored(f"Tool output:\n{output}", 'green'))
            tool_output_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call['id']))

        return {"messages": state["messages"] + [response] + tool_output_messages}


# ==============================================================================
#
# CRITIC NODE
#
# ==============================================================================

class CriticNode:
    """The node responsible for evaluating the agent's responses."""
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an AI critic. Your job is to evaluate an agent's response to a user query based on the provided conversation history.

Critique Criteria:
- **Accuracy**: Is the information correct?
- **Completeness**: Does the response fully answer the user's question?
- **Clarity**: Is the answer clear and not just raw tool output?

Instructions:
- Begin your critique with one of two words: `REVISE` or `ACCEPT`.
- If the agent's last response is not a complete, user-friendly answer (e.g., it's raw JSON or a tool call message), you MUST respond with `REVISE`.
- If the agent's answer is insufficient, respond with `REVISE` and provide a concise, one-sentence explanation of what is missing or wrong.
- If the agent's answer is satisfactory, respond with `ACCEPT`.
"""),
            ("human", "Conversation History:\n{conversation_history}\n\nYour Critique:")
        ])

    def run(self, state: AgentState):
        """The main execution method for the critic node."""
        history_str = "\n".join([f"{type(msg).__name__}: {msg.content}" for msg in state["messages"]])
        
        chain = self.prompt_template | self.llm
        response = chain.invoke({"conversation_history": history_str})
        critique = response.content.strip()

        if debug_mode:
            print(colored(f"Critic response:\n{critique}", 'yellow'))

        return {
            "critique": critique,
            "iterations": state["iterations"] + 1
        }


# ==============================================================================
#
# GRAPH CONDITIONAL LOGIC
#
# ==============================================================================

def should_continue(state: AgentState) -> str:
    """Determines the next step in the workflow based on the critic's feedback."""
    if state["iterations"] > 5:
        print(colored("--- Too many iterations, ending workflow ---", "red"))
        return END
    
    if state.get("critique", "").strip().upper().startswith("REVISE"):
        return "agent"
    else:
        return END


# ==============================================================================
#
# GRAPH COMPILATION (Moved to Global Scope)
#
# The following logic is moved to the global scope. This creates the compiled
# `app` object when the module is first imported, making it available for other
# scripts (like evaluate.py) to import and use.
#
# ==============================================================================

# 1. Instantiate the nodes
agent_node = AgentNode(llm=agent_llm, tools=tools)
critic_node = CriticNode(llm=critic_llm)

# 2. Define the graph structure
graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node.run)
graph_builder.add_node("critic", critic_node.run)

# 3. Define the workflow edges
graph_builder.set_entry_point("agent")
graph_builder.add_edge("agent", "critic")
graph_builder.add_conditional_edges(
    "critic",
    should_continue,
    {
        "agent": "agent",
        END: END
    }
)

# 4. Compile the graph into a runnable application
app = graph_builder.compile()


# ==============================================================================
#
# MAIN EXECUTION BLOCK (for interactive chat)
#
# ==============================================================================

def main():
    """
    Runs the interactive command-line chat application.
    This function is only called when `agent.py` is run directly.
    """
    print(colored("Welcome to LangGraph interactive AI Agent!", "cyan"))
    print("Type your question (or 'exit' to quit):")

    while True:
        q = input(colored("Your question: ", "green")).strip()
        if not q or q.lower() == "exit":
            print(colored("Goodbye!", "cyan"))
            break

        initial_state: AgentState = {
            "messages": [HumanMessage(content=q)],
            "critique": "",
            "iterations": 0,
        }

        # The 'app' object is now defined globally, so we can use it here.
        final_state = app.invoke(initial_state)

        print(colored("\n--- Final Answer ---", "cyan"))
        final_answer = final_state["messages"][-1]
        
        if isinstance(final_answer, ToolMessage):
             final_answer = next((m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage) and not m.tool_calls), None)

        if final_answer:
            print(colored(final_answer.content, "white"))
        else:
            print(colored("The agent could not produce a final answer.", "red"))

        if debug_mode:
            print(colored("\n--- Debug Info ---", "yellow"))
            print(colored(f"Final Critique: {final_state.get('critique', '')}", "yellow"))
            print(colored(f"Total Iterations: {final_state['iterations']}", "yellow"))
        print()

# This standard Python construct ensures that the main() function is called
# only when the script is executed directly.
if __name__ == "__main__":
    main()
