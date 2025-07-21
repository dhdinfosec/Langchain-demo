import os
import time
from dotenv import load_dotenv
from typing import List, Annotated, TypedDict
import operator
# Removed local model specific imports, as per previous updates:
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
# import torch

# Import Google Generative AI components for LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# --- Load environment variables ---
load_dotenv()

# --- LangSmith Configuration (Optional but Recommended) ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LangGraph Agent with Critic")

# For debugging environment variables
print(f"DEBUG: LANGCHAIN_TRACING_V2: {os.getenv('LANGCHAIN_TRACING_V2')}")
print(f"DEBUG: LANGCHAIN_API_KEY (first 5 chars): {os.getenv('LANGCHAIN_API_KEY')[:5] if os.getenv('LANGCHAIN_API_KEY') else 'None'}")
print(f"DEBUG: LANGCHAIN_PROJECT: {os.getenv('LANGCHAIN_PROJECT')}")
print(f"DEBUG: GOOGLE_API_KEY (first 5 chars): {os.getenv('GOOGLE_API_KEY')[:5] if os.getenv('GOOGLE_API_KEY') else 'None'}")


# --- Model Configuration ---
# Use specific Gemini models from your ListModels.txt output
AGENT_LLM_MODEL_NAME = "gemini-1.5-flash-latest" # Faster, cost-effective for agent's initial thoughts/tooling
CRITIC_LLM_MODEL_NAME = "gemini-1.5-pro-latest"  # More capable for nuanced critique

# Initialize Gemini LLMs
print(f"\n--- Initializing Gemini Models ---")
start_model_load_time = time.perf_counter()

# Ensure GOOGLE_API_KEY is set in your environment or .env file
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it in your .env file.")

print(f"Connecting to Agent LLM: {AGENT_LLM_MODEL_NAME}...")
agent_llm = ChatGoogleGenerativeAI(model=AGENT_LLM_MODEL_NAME, temperature=0.0, google_api_key=google_api_key)
print(f"INFO: Agent LLM ({AGENT_LLM_MODEL_NAME}) initialized.")

print(f"Connecting to Critic LLM: {CRITIC_LLM_MODEL_NAME}...")
critic_llm = ChatGoogleGenerativeAI(model=CRITIC_LLM_MODEL_NAME, temperature=0.0, google_api_key=google_api_key)
print(f"INFO: Critic LLM ({CRITIC_LLM_MODEL_NAME}) initialized.")

end_model_load_time = time.perf_counter()
print(f"INFO: All Gemini API clients initialized in {end_model_load_time - start_model_load_time:.2f} seconds.")


# --- Define Tools (Example: Wikipedia Tool) ---
@tool
def wikipedia_search(query: str) -> str:
    """Searches Wikipedia for the given query and returns a summary."""
    try:
        import wikipedia
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Error performing Wikipedia search: {e}"

tools = [wikipedia_search]

# --- Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    tool_output: str
    critique: str
    iterations: int

# --- Agent Node ---
class AgentNode:
    def __init__(self, agent_llm, tools):
        self.agent_llm = agent_llm
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.tools_list = tools # Keep original list for prompt formatting

        # Define the agent's prompt template once
        self.agent_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. You have access to the following tools:
{tool_names_with_descriptions}

**Instructions for Tool Usage:**
1. If the user's query requires external, up-to-date, or specific factual information (e.g., "current president", "population", "details about X"), you MUST use a tool to get the information.
2. To use a tool, your response MUST be a JSON object in the exact format:
   {{
       "tool": "tool_name",
       "tool_input": "input for the tool"
   }}
3. If you use a tool, DO NOT add any other text before or after the JSON.
4. If you have the information from a tool call (or if the query can be answered from your internal knowledge and does NOT require a tool), respond directly to the user.

**Instructions for Direct Answers (No Tool):**
1. Your direct answer should be concise and to the point.
2. DO NOT include any prefixes like "AI:", "Agent:", "Response:", etc. Just the answer.
3. DO NOT ask follow-up questions or try to continue the conversation.
4. Respond only to the current user query.

**Example of Tool Usage:**
User: Search Wikipedia for 'Python programming language'
Assistant: {{"tool": "wikipedia_search", "tool_input": "Python programming language"}}
"""),
            HumanMessage(content="{user_query}")
        ])

    def run(self, state: AgentState) -> AgentState:
        print("\n---AGENT NODE (Thinking...)---")
        start_agent_inference_time = time.perf_counter()
        messages = state["messages"]
        user_query = messages[0].content # Original user query is the first message

        # Format the prompt with dynamic tool descriptions and user query
        formatted_messages = self.agent_prompt_template.partial(
            tool_names_with_descriptions="\n".join([f"- {t.name}: {t.description}" for t in self.tools_list])
        ).format_messages(user_query=user_query)


        print("\nDEBUG: Agent LLM Prompt (messages sent to model):")
        for msg in formatted_messages:
            print(f"{type(msg).__name__}: {msg.content}")
        print("---END PROMPT---")

        # Invoke the Gemini LLM
        response_message = self.agent_llm.invoke(formatted_messages)
        generated_text = response_message.content # Extract content from AIMessage

        print(f"DEBUG: Raw generated text from Agent LLM: \n---\n{generated_text}\n---")

        tool_call = None
        # Attempt to parse tool call from the generated_text
        if generated_text.strip().startswith("{") and generated_text.strip().endswith("}"):
            try:
                import json
                tool_call = json.loads(generated_text.strip())
                if not isinstance(tool_call, dict) or "tool" not in tool_call or "tool_input" not in tool_call:
                    tool_call = None # Not a valid tool call format
            except (json.JSONDecodeError, SyntaxError) as e:
                print(f"Warning: Could not parse JSON tool call from agent response: {e}. Response was: '{generated_text}'")
                tool_call = None
        else:
            tool_call = None # Not a JSON, so it's a direct answer

        if tool_call and tool_call.get("tool") in self.tools_by_name:
            print(f"DECISION: Agent decided to use tool: {tool_call['tool']}")
            tool_name = tool_call["tool"]
            tool_input = tool_call["tool_input"]
            print(f"Calling tool: {tool_name} with input: '{tool_input}'")
            tool_func = self.tools_by_name[tool_name]
            output = tool_func.invoke(tool_input)
            print(f"Tool output: {output}")
            end_agent_inference_time = time.perf_counter()
            print(f"INFO: Agent Node execution took {end_agent_inference_time - start_agent_inference_time:.2f} seconds.")
            # Record tool use in messages
            return {"messages": messages + [AIMessage(content=f"Tool Used: {tool_name} with input '{tool_input}'.\nOutput: {output}")], "tool_output": output}
        else:
            print("DECISION: Agent decided to provide a direct answer (no tool usage).")
            final_agent_response = generated_text
            print(f"Agent Final Response (for this turn): {final_agent_response}")
            end_agent_inference_time = time.perf_counter()
            print(f"INFO: Agent Node execution took {end_agent_inference_time - start_agent_inference_time:.2f} seconds.")
            return {"messages": messages + [AIMessage(content=final_agent_response)], "tool_output": ""}

# --- Critic Node ---
class CriticNode:
    def __init__(self, critic_llm):
        self.critic_llm = critic_llm

        # Define the critic's prompt template once
        self.critique_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an AI critic assisting another AI agent. Your goal is to evaluate the agent's response to the user's query.
Consider the original user query and the agent's latest response.

**Evaluation Criteria:**
1.  **Relevance:** Does the agent's response directly address the user's query?
2.  **Accuracy:** Is the information provided accurate and free of hallucinations?
3.  **Completeness:** Is the response sufficiently detailed or does it leave out important information?
4.  **Clarity:** Is the response easy to understand?
5.  **Tool Usage:** If a tool was used, was it appropriate and was its output integrated effectively?

Provide a concise critique based on these criteria.
If the agent's response needs *significant* improvement or refinement, explicitly state 'REVISE' at the beginning of your critique.
Otherwise, if the response is good or only needs minor tweaks, state 'ACCEPT'.
**IMPORTANT: Provide only the critique. Do not generate conversational filler or follow-up questions. Your response should start directly with 'REVISE' or 'ACCEPT'.**
"""),
            HumanMessage(content="Original User Query:\n{user_query}\n\nAgent's Latest Response:\n{agent_response}\n\nYour Critique (Start with REVISE or ACCEPT):")
        ])

    def run(self, state: AgentState) -> AgentState:
        print("\n---CRITIC NODE (Reviewing...)---")
        start_critic_inference_time = time.perf_counter()
        messages = state["messages"]
        last_agent_message = messages[-1]
        user_query = messages[0].content # Original user query

        # Format the critic prompt
        critique_messages = self.critique_prompt_template.format_messages(
            user_query=user_query,
            agent_response=last_agent_message.content
        )

        print("\nDEBUG: Critic LLM Prompt (messages sent to model):")
        for msg in critique_messages:
            print(f"{type(msg).__name__}: {msg.content}")
        print("---END PROMPT---")

        critique_response_message = self.critic_llm.invoke(critique_messages)
        critique_text = critique_response_message.content

        print(f"Critic's Critique: {critique_text}")
        end_critic_inference_time = time.perf_counter()
        print(f"INFO: Critic Node execution took {end_critic_inference_time - start_critic_inference_time:.2f} seconds.")
        return {"critique": critique_text, "iterations": state["iterations"] + 1}

# --- Conditional Edge for Critic Feedback ---
def should_continue(state: AgentState) -> str:
    # Ensure critique text starts with "REVISE" to trigger revision
    if state["critique"].strip().upper().startswith("REVISE"):
        print("\n---DECISION: REVISE (Agent will try again)---")
        if state["iterations"] < 3: # Max 3 attempts (initial + 2 revisions)
            return "revise"
        else:
            print("\n---DECISION: Max revision attempts reached. Ending.---")
            return "end"
    else:
        print("\n---DECISION: ACCEPT (Response finalized)---")
        return "end"

# --- Build the LangGraph Workflow ---
workflow = StateGraph(AgentState)

# Initialize nodes with the new LLM instances
agent_node_instance = AgentNode(agent_llm, tools)
critic_node_instance = CriticNode(critic_llm)

workflow.add_node("agent", agent_node_instance.run)
workflow.add_node("critic", critic_node_instance.run)

workflow.set_entry_point("agent")
workflow.add_edge("agent", "critic")
workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "revise": "agent",
        "end": END
    }
)

app = workflow.compile()

# --- Interactive Usage ---
if __name__ == "__main__":
    print("\n--- Welcome to the LangGraph Agent Demo (Gemini API)! ---")
    print("Type your questions below. Type 'exit' to quit.")

    while True:
        user_input = input("\n> You: ")
        if user_input.lower() == 'exit':
            print("Exiting demo. Goodbye!")
            break

        initial_state = {"messages": [HumanMessage(content=user_input)], "tool_output": "", "critique": "", "iterations": 0}

        final_answer = None
        for s in app.stream(initial_state):
            # Print intermediate state changes for debugging (optional)
            # print(s)
            # print("---")
            if "__end__" in s:
                # The final_answer might be the agent's last direct response
                # or the tool output if the critic accepted it.
                # For simplicity, we'll take the last message in the stream state.
                final_answer_message = s["__end__"].get("messages", [])
                if final_answer_message:
                    final_answer = final_answer_message[-1].content
                break # Break after the graph ends

        if final_answer:
            print(f"\n> Agent: {final_answer}")
        else:
            print("\n> Agent: I'm sorry, I couldn't process that request or provide a final answer.")
