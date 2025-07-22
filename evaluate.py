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
# 2. It uses a predefined evaluation dataset containing questions and expected answers.
# 3. It iterates through each question, invokes the agent, and captures the
#    final response.
# 4. It automatically generates keywords from the expected answer and checks if all
#    keywords are present in the agent's response.
# 5. Finally, it prints a summary report detailing the pass/fail results.
#
# ==============================================================================

# --- Core Imports ---
import sys
import re
import logging

# --- Third-party Library Imports ---
# termcolor: For adding colored text to the terminal for readability.
try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs): return text
    logging.warning("'termcolor' not found; install for colored output.")

# --- Local Application Imports ---
# This is the most crucial part: we import the compiled agent application `app`
# and necessary data classes directly from your `agent.py` file.
# This allows us to test the agent without duplicating its code.
from agent import app, AgentState, HumanMessage, AIMessage, ToolMessage


# ==============================================================================
#
# EVALUATION TEST SUITE
#
# This section defines the set of questions and corresponding answers that will
# be used to test the agent, based on the provided list.
#
# ==============================================================================

EVALUATION_DATASET = [
    {"inputs": {"question": "What is the capital of Japan?"}, "outputs": {"answer": "Tokyo"}},
    {"inputs": {"question": "Who invented the telephone?"}, "outputs": {"answer": "Alexander Graham Bell"}},
    {"inputs": {"question": "What is the tallest mountain in the world?"}, "outputs": {"answer": "Mount Everest"}},
    {"inputs": {"question": "What is Java? (programming language)"}, "outputs": {"answer": "Java is a high-level, class-based, object-oriented programming language."}},
    {"inputs": {"question": "When was the Eiffel Tower built?"}, "outputs": {"answer": "1889"}},
    {"inputs": {"question": "What is the population of Canada?"}, "outputs": {"answer": "Approximately 38 million"}},
    {"inputs": {"question": "Who wrote Hamlet?"}, "outputs": {"answer": "William Shakespeare"}},
    {"inputs": {"question": "What is the chemical symbol for gold?"}, "outputs": {"answer": "Au"}},
]


# ==============================================================================
#
# MAIN EXECUTION BLOCK
#
# This function orchestrates the entire evaluation process, from running the
# tests to printing the final summary.
#
# ==============================================================================

def main():
    """
    Runs the automated evaluation process against the agent.
    """
    print(colored("--- Starting Agent Evaluation ---", "cyan"))

    # Initialize counters to track test results.
    passed_count = 0
    failed_count = 0
    # A list to store detailed information about failed tests for the final report.
    failure_details = []

    # Loop through each test case in our predefined dataset.
    for i, test_case in enumerate(EVALUATION_DATASET):
        query = test_case["inputs"]["question"]
        expected_answer = test_case["outputs"]["answer"]

        # To make the check robust, we automatically generate keywords from the
        # expected answer string. This avoids issues with slightly different phrasing.
        # It removes common punctuation, lowercases, and splits the answer into words.
        cleaned_answer = re.sub(r'[(),.]', '', expected_answer)
        expected_keywords = cleaned_answer.lower().split()

        print(colored(f"\n[{i+1}/{len(EVALUATION_DATASET)}] Running test for query: '{query}'", "white"))

        # For each new question, create a fresh initial state for the graph.
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "critique": "",
            "iterations": 0,
        }

        # Invoke the compiled agent graph with the initial state.
        final_state = app.invoke(initial_state)

        # Extract the final answer message from the conversation history.
        # This logic ensures we get the last user-facing AI response.
        final_answer_msg = final_state["messages"][-1]
        if isinstance(final_answer_msg, ToolMessage):
             final_answer_msg = next((m for m in reversed(final_state["messages"]) if isinstance(m, AIMessage) and not m.tool_calls), None)

        # --- Evaluate the result ---
        test_passed = False
        agent_answer = ""
        if final_answer_msg and final_answer_msg.content:
            agent_answer = final_answer_msg.content
            # Check if all generated keywords are present in the agent's answer (case-insensitive).
            if all(keyword in agent_answer.lower() for keyword in expected_keywords):
                test_passed = True

        # --- Record and report the result for this test case ---
        if test_passed:
            passed_count += 1
            print(colored("  [PASS]", "green"))
        else:
            failed_count += 1
            print(colored("  [FAIL]", "red"))
            # Store the details of the failure for the summary report.
            failure_details.append({
                "query": query,
                "expected_answer": expected_answer,
                "actual_answer": agent_answer if agent_answer else "No answer produced."
            })

    # --- Print Final Summary Report ---
    print(colored("\n\n--- Evaluation Summary ---", "cyan"))
    print(colored(f"Total Tests: {len(EVALUATION_DATASET)}", "white"))
    print(colored(f"  Passed: {passed_count}", "green"))
    print(colored(f"  Failed: {failed_count}", "red"))

    # If there were any failures, print the details to help with debugging.
    if failure_details:
        print(colored("\n--- Failure Details ---", "yellow"))
        for i, failure in enumerate(failure_details):
            print(colored(f"\n{i+1}. Failed Query:", "yellow") + f" {failure['query']}")
            print(colored("  - Expected Answer:", "yellow") + f" {failure['expected_answer']}")
            print(colored("  - Actual Answer:", "yellow") + f" {failure['actual_answer']}")

# This standard Python construct ensures that the main() function is called
# only when the script is executed directly.
if __name__ == "__main__":
    # Add a check to ensure the script is not run with the debug flag,
    # as the output can be overwhelming for an automated evaluation.
    if '--debug' in sys.argv:
        print(colored("ERROR: This script is intended for automated evaluation and should not be run with the --debug flag.", "red"))
        sys.exit(1)
    main()
