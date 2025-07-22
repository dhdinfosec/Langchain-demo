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
# Key Components:
# 1. AgentNode: The "worker" AI that understands the user's query and uses
#    tools to find information.
# 2. CriticNode: The "evaluator" AI that checks the agent's work and provides
#    feedback.
# 3. StateGraph: A LangGraph object that defines the flow of logic between the
#    agent and the critic.
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
# python-dotenv: For loading environment variables from a .env file (e.g., API keys)
from dotenv import load_dotenv

# LangChain and LangGraph: The core frameworks for building the agent and graph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# termcolor: Optional library for adding colored text to the terminal for readability
try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs): return text
    logging.warning("'termcolor' not found; install for colored output.")


# ==============================================================================
#
# SCRIPT SETUP AND CONFIGURATION
#
# This section handles initial setup tasks:
# - Parsing command-line arguments for debug mode.
# - Configuring logging to save debug and info messages to a file.
# - Loading environment variables from the .env file.
# - Setting up LangSmith tracing for observability.
#
# ==============================================================================

# Check for a '--debug' flag when running the script to enable verbose logging
debug_mode = '--debug' in sys.argv
if debug_mode:
    print(colored("DEBUG MODE ENABLED: verbose logging active.", 'yellow'))
    logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(filename='debug.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables (e.g., GOOGLE_API_KEY) from a file named '.env'
load_dotenv()

# Configure LangChain settings for LangSmith tracing, which helps in debugging and monitoring the agent's behavior
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LangGraph Agent with Critic")

# Define which Google Gemini models to use for the agent and the critic
AGENT_MODEL = "gemini-2.0-flash"
CRITIC_MODEL = "gemini-2.5-flash"

# Retrieve the Google API key from environment variables and raise an error if not found
google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    raise RuntimeError("GOOGLE_API_KEY not set in environment or .env file.")


# ==============================================================================
#
# MODELS AND TOOLS INITIALIZATION
#
# This section defines the AI models and the tools they can use.
# - Two instances of ChatGoogleGenerativeAI are created for the agent and critic.
# - A custom tool for searching Wikipedia is defined.
#
# ==============================================================================

# Initialize the LLM for the agent. Temperature=0 makes its responses more deterministic.
agent_llm = ChatGoogleGenerativeAI(model=AGENT_MODEL, temperature=0, google_api_key=google_key)
# Initialize the LLM for the critic.
critic_llm = ChatGoogleGenerativeAI(model=CRITIC_MODEL, temperature=0, google_api_key=google_key)

@tool("wikipedia_search", description="Search Wikipedia and return a summary.")
def wikipedia_search(query: str) -> str:
    """
    A tool that allows the agent to search Wikipedia.

    It takes a search query, finds the relevant Wikipedia page, and returns a
    summary of the first few sentences. It includes error handling for common
    issues like pages not being found or ambiguous queries.

    Args:
        query (str): The search term for Wikipedia.

    Returns:
        str: A summary of the Wikipedia page or an error message.
    """
    try:
        import wikipedia
        if debug_mode:
            print(colored(f"DEBUG: Wikipedia search for '{query}'", "yellow"))
        # First, try a direct search without auto-suggestion for accuracy.
        return wikipedia.summary(query, sentences=5, auto_suggest=False)
    except wikipedia.exceptions.PageError:
        if debug_mode:
            print(colored(f"DEBUG: PageError, retry with auto_suggest for '{query}'", "yellow"))
        try:
            # If the direct search fails, fallback to auto_suggest to find a likely match.
            return wikipedia.summary(query, sentences=5, auto_suggest=True)
        except Exception as e:
            return f"Wikipedia fallback error: {e}"
    except wikipedia.exceptions.DisambiguationError as e:
        # If the query could refer to multiple pages, inform the agent of the options.
        return f"Wikipedia disambiguation error for query '{query}': Options: {e.options[:5]}"
    except Exception as e:
        return f"Wikipedia search error: {e}"

# Create a list of tools that the agent can use.
tools = [wikipedia_search]


# ==============================================================================
#
# GRAPH STATE DEFINITION
#
# The AgentState class defines the "memory" of our graph. It's a dictionary
# that holds all the information passed between the nodes (agent and critic).
# Each field represents a piece of the shared state.
#
# ==============================================================================

class AgentState(TypedDict):
    """
    Represents the state of the agent workflow. This state is passed between
    nodes in the graph.

    Attributes:
        messages (List[BaseMessage]): The history of the conversation, including
            human queries, AI responses, and tool outputs.
        critique (str): The feedback provided by the critic node after evaluating
            the agent's response.
        iterations (int): A counter for the number of loops (agent -> critic)
            to prevent infinite cycles.
    """
    messages: List[BaseMessage]
    critique: str
    iterations: int


# ==============================================================================
#
# AGENT NODE
#
# The AgentNode is the primary "worker" of the system. Its main responsibility
# is to process the user's request, decide whether to use a tool, call that
# tool if necessary, and formulate a response.
#
# ==============================================================================

class AgentNode:
    """
    The node responsible for executing the main agent logic. It generates
    responses and decides when to use tools.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI, tools: List[tool]):
        """
        Initializes the AgentNode.

        Args:
            llm (ChatGoogleGenerativeAI): The language model for the agent.
            tools (List[tool]): A list of tools the agent is allowed to use.
        """
        # Binding tools to the LLM allows the model to natively understand
        # and request tool calls in its preferred format.
        self.llm = llm.bind_tools(tools)
        self.tools = {t.name: t for t in tools}
        today = datetime.now().strftime("%Y-%m-%d")

        # This system prompt is the agent's core instruction set. It defines its
        # persona, knowledge, and rules of engagement.
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
        """
        The main execution method for the agent node.

        Args:
            state (AgentState): The current state of the workflow.

        Returns:
            dict: A dictionary with the updated state.
        """
        if debug_mode:
            print(colored(f"\nAGENT NODE ITERATION {state['iterations'] + 1}", 'magenta'))
            if state.get("critique"):
                print(colored(f"Critique: {state['critique']}", 'yellow'))

        # Create a runnable chain from the prompt and LLM.
        # This chain will format the prompt with the current state and pass it to the model.
        chain = self.prompt_template | self.llm
        response = chain.invoke({
            "critique": state.get("critique", ""),
            "user_query": state["messages"][0].content,
            "messages": state["messages"]
        })

        # If the model's response does not contain a tool call, it's a direct
        # answer, and we can return it.
        if not response.tool_calls:
            if debug_mode:
                print(colored("Agent provides direct answer.", 'cyan'))
            return {"messages": state["messages"] + [response]}

        if debug_mode:
            print(colored("Agent requested tool call.", 'cyan'))
        
        # If the model requested a tool call, process it here.
        tool_output_messages = []
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_input = tool_call["args"]
            if debug_mode:
                print(colored(f"Calling tool '{tool_name}' with input {tool_input}", 'cyan'))

            # Invoke the correct tool with the arguments provided by the model.
            try:
                output = self.tools[tool_name].invoke(tool_input)
            except TypeError:
                # This fallback handles cases where the model might pass a dict
                # with one key, but the tool expects a single string argument.
                output = self.tools[tool_name].invoke(next(iter(tool_input.values())))

            if debug_mode:
                print(colored(f"Tool output:\n{output}", 'green'))
            
            # Create a ToolMessage with the output of the tool call. This message
            # will be added to the conversation history so the agent knows what
            # the tool returned.
            tool_output_messages.append(ToolMessage(content=str(output), tool_call_id=tool_call['id']))

        # Return the updated messages, including the AI's tool request and the tool's output.
        return {"messages": state["messages"] + [response] + tool_output_messages}


# ==============================================================================
#
# CRITIC NODE
#
# The CriticNode acts as the quality control for the agent. It reviews the
# conversation and the agent's latest response, providing feedback on whether
# the answer is acceptable or needs revision.
#
# ==============================================================================

class CriticNode:
    """
    The node responsible for evaluating the agent's responses.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI):
        """
        Initializes the CriticNode.

        Args:
            llm (ChatGoogleGenerativeAI): The language model for the critic.
        """
        # The critic's system prompt defines its role and the criteria for evaluation.
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

Example:
User Query: Who is the president of the USA?
Agent Response: Tool Used: wikipedia_search with input 'current US president'. Output: Joseph R. Biden Jr. is the 46th and current president...
Your Critique: REVISE. The agent provided raw tool output instead of a clean answer.
"""),
            ("human", "Conversation History:\n{conversation_history}\n\nYour Critique:")
        ])

    def run(self, state: AgentState):
        """
        The main execution method for the critic node.

        Args:
            state (AgentState): The current state of the workflow.

        Returns:
            dict: A dictionary containing the critique and updated iteration count.
        """
        # Combine the entire message history into a single string for the critic to review.
        history_str = "\n".join([f"{type(msg).__name__}: {msg.content}" for msg in state["messages"]])
        
        # Invoke the critic model.
        chain = self.prompt_template | self.llm
        response = chain.invoke({"conversation_history": history_str})
        critique = response.content.strip()

        if debug_mode:
            print(colored(f"Critic response:\n{critique}", 'yellow'))

        # Return the critique to be added to the state. The agent will see this
        # in the next iteration.
        return {
            "critique": critique,
            "iterations": state["iterations"] + 1
        }

# ==============================================================================
#
# GRAPH CONDITIONAL LOGIC
#
# This function defines the control flow of the graph. After the critic has
# run, this function decides whether to loop back to the agent for revisions
# or to end the workflow.
#
# ==============================================================================

def should_continue(state: AgentState) -> str:
    """
    Determines the next step in the workflow based on the critic's feedback.

    - If the critic says `REVISE`, the graph transitions back to the 'agent' node.
    - If the critic says `ACCEPT`, the graph transitions to the `END` state.
    - If the loop has run too many times, it ends to prevent infinite cycles.

    Args:
        state (AgentState): The current state of the workflow.

    Returns:
        str: The name of the next node to execute ('agent' or END).
    """
    if state["iterations"] > 5: # Increased limit for more complex queries
        print(colored("--- Too many iterations, ending workflow ---", "red"))
        return END
    
    # Check the critique from the state to decide the next step.
    if state.get("critique", "").strip().upper().startswith("REVISE"):
        return "agent"
    else:
        return END

# ==============================================================================
#
# MAIN EXECUTION BLOCK
#
# This is the entry point of the script. It sets up the graph, wires the nodes
# and logic together, and starts the interactive chat loop.
#
# ==============================================================================

def main():
    """
    Sets up the LangGraph graph and runs the interactive command-line chat application.
    """
    # 1. Instantiate the nodes
    agent = AgentNode(llm=agent_llm, tools=tools)
    critic = CriticNode(llm=critic_llm)
    
    # 2. Define the graph structure
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", agent.run)
    graph_builder.add_node("critic", critic.run)
    
    # 3. Define the workflow edges
    graph_builder.set_entry_point("agent")
    graph_builder.add_edge("agent", "critic")
    graph_builder.add_conditional_edges(
        "critic",
        should_continue,
        # The dictionary maps the output of should_continue to the next node
        {
            "agent": "agent",
            END: END
        }
    )
    
    # 4. Compile the graph into a runnable application
    app = graph_builder.compile()

    # 5. Start the interactive chat loop
    print(colored("Welcome to LangGraph interactive AI Agent!", "cyan"))
    print("Type your question (or 'exit' to quit):")

    while True:
        q = input(colored("Your question: ", "green")).strip()
        if not q or q.lower() == "exit":
            print(colored("Goodbye!", "cyan"))
            break

        # For each new question, create a fresh initial state
        initial_state: AgentState = {
            "messages": [HumanMessage(content=q)],
            "critique": "",
            "iterations": 0,
        }

        # Invoke the graph with the initial state and stream the results
        final_state = app.invoke(initial_state)

        # Print the final, user-friendly answer
        print(colored("\n--- Final Answer ---", "cyan"))
        final_answer = final_state["messages"][-1]
        
        # The last message might be a ToolMessage. If so, find the last actual
        # AI message to display as the final answer.
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
