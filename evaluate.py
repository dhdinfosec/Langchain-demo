# ==============================================================================
#
# Automated Evaluation Script for the LangGraph Agent
#
# This script is designed to test the conversational agent defined in `agent.py`.
# It runs a predefined suite of test questions against the agent and evaluates
# the responses automatically.
#
# How it works:
# 1. It imports the compiled agent application (`app`) from `agent.py`.
# 2. It uses a predefined evaluation dataset containing questions and a list of
#    essential keywords expected in the answer.
# 3. It iterates through each question, invokes the agent, and captures the
#    final response.
# 4. It checks if all expected keywords are present in the agent's answer.
# 5. Finally, it prints a summary report detailing the pass/fail results.
#
# ==============================================================================

# --- Core Imports ---
import sys
import re
import logging
from datetime import datetime

# --- Third-party Library Imports ---
# termcolor is used for adding colored text to the terminal for better readability.
try:
    from termcolor import colored
except ImportError:
    # If termcolor is not installed, define a fallback function so the script doesn't crash.
    def colored(text, *args, **kwargs): return text
    logging.warning("'termcolor' not found; install for colored output.")

# --- Local Application Imports ---
# We import the 'agent' module itself to control its debug mode, along with
# the compiled 'app' and necessary data classes from our main agent script.
import agent
from agent import app, AgentState, HumanMessage, AIMessage, ToolMessage

# --- Debug Mode Configuration ---
# Set the debug mode on the imported agent module. This activates the
# verbose print statements inside the agent's logic if the script is run
# with the --debug flag (e.g., `python3 evaluate.py --debug`).
agent.debug_mode = '--debug' in sys.argv

# ==============================================================================
#
# EVALUATION TEST SUITE
#
# This section defines the test cases for the agent. Each case consists of a
# query and a list of essential, case-insensitive keywords.
#
# This keyword-based approach makes the tests robust. It allows the agent's
# answer to have different phrasing or more detail, as long as it contains the
# core facts we are looking for.
#
# ==============================================================================

EVALUATION_DATASET = [
    {"query": "What is the capital of Japan?", "expected_keywords": ["tokyo"]},
    {"query": "Who invented the telephone?", "expected_keywords": ["alexander", "graham", "bell"]},
    {"query": "What is the tallest mountain in the world?", "expected_keywords": ["mount", "everest"]},
    {"query": "What is Java? (programming language)", "expected_keywords": ["object-oriented", "programming", "language"]},
    {"query": "When was the Eiffel Tower built?", "expected_keywords": ["1889"]},
    # For facts that change over time (like population), keywords are chosen to be flexible.
    {"query": "What is the population of Canada?", "expected_keywords": ["million", "canada"]},
    {"query": "Who wrote Hamlet?", "expected_keywords": ["william", "shakespeare"]},
    {"query": "What is the chemical symbol for gold?", "expected_keywords": ["au"]},
]


# ==============================================================================
#
# MAIN EXECUTION BLOCK
#
# This function orchestrates the entire evaluation process, from running the
# tests to printing the final summary report.
#
# ==============================================================================

def main():
    """
    Runs the automated evaluation process against the agent.
    """
    print(colored("--- Starting Agent Evaluation ---", "cyan"))
    if agent.debug_mode:
        print(colored("Debug mode enabled. Agent will provide verbose output.", "yellow"))

    # Initialize counters to track test results.
    passed_count = 0
    failed_count = 0
    # A list to store detailed information about failed tests for the final report.
    failure_details = []

    # Get the current date once before the loop starts to ensure consistency across tests.
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Loop through each test case in our predefined dataset.
    for i, test_case in enumerate(EVALUATION_DATASET):
        query = test_case["query"]
        expected_keywords = test_case["expected_keywords"]

        print(colored(f"\n[{i+1}/{len(EVALUATION_DATASET)}] Running test for query: '{query}'", "white"))

        # For each new question, create a fresh initial state for the graph.
        # This must match the AgentState definition in agent.py.
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "critique": "",
            "iterations": 0,
            "today": today_str,
            "step": 1,
        }

        # Invoke the compiled agent graph with the initial state.
        final_state = app.invoke(initial_state)

        # Extract the final answer message from the conversation history.
        # The last message might be a ToolMessage, so we find the last AIMessage instead.
        final_answer_msg = final_state["messages"][-1]
        if isinstance(final_answer_msg, ToolMessage):
             final_answer_msg = next((m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage) and not m.tool_calls), None)

        # Extract the text content of the agent's answer.
        agent_answer = ""
        if final_answer_msg and final_answer_msg.content:
            agent_answer = final_answer_msg.content
        
        # Print the agent's actual answer for every test for immediate feedback.
        print(colored(f"  - Agent's Answer: {agent_answer}", "white"))

        # Evaluate the result based on the keywords.
        test_passed = False
        if agent_answer:
            # Check if all essential keywords are present in the agent's answer.
            if all(keyword.lower() in agent_answer.lower() for keyword in expected_keywords):
                test_passed = True

        # Record and report the result for this specific test case.
        if test_passed:
            passed_count += 1
            print(colored("  [PASS]", "green"))
        else:
            failed_count += 1
            print(colored("  [FAIL]", "red"))
            # Store the details of the failure for the summary report.
            failure_details.append({
                "query": query,
                "expected_keywords": expected_keywords,
                "actual_answer": agent_answer if agent_answer else "No answer produced."
            })

    # Print the final summary report after all tests are complete.
    print(colored("\n\n--- Evaluation Summary ---", "cyan"))
    print(colored(f"Total Tests: {len(EVALUATION_DATASET)}", "white"))
    print(colored(f"  Passed: {passed_count}", "green"))
    print(colored(f"  Failed: {failed_count}", "red"))

    # If there were any failures, print the details to help with debugging.
    if failure_details:
        print(colored("\n--- Failure Details ---", "yellow"))
        for i, failure in enumerate(failure_details):
            print(colored(f"\n{i+1}. Failed Query:", "yellow") + f" {failure['query']}")
            print(colored("  - Expected Keywords:", "yellow") + f" {failure['expected_keywords']}")
            print(colored("  - Actual Answer:", "yellow") + f" {failure['actual_answer']}")

# This standard Python construct ensures that the main() function is called
# only when the script is executed directly.
if __name__ == "__main__":
    main()
