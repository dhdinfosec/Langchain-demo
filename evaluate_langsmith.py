# ==============================================================================
#
# LangSmith SDK Evaluation Script (Using Existing UI Dataset)
#
# This script runs an evaluation against a dataset in LangSmith.
#
# ==============================================================================

# --- Core Imports ---
import os
import sys
from datetime import datetime

# --- Third-party Library Imports ---
from dotenv import load_dotenv
load_dotenv(override=True)  # Override any cached env vars
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.messages import HumanMessage
from termcolor import colored

# --- Local Application Imports ---
from agent import app, AgentState

# ==============================================================================
#
# SCRIPT CONFIGURATION
#
# ==============================================================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LangGraph Wikipedia Agent - Evaluation")

# ==============================================================================
#
# AGENT INVOCATION LOGIC
#
# ==============================================================================
def run_agent(props: dict):
    """The target function for the evaluation, running the agent for a single data point."""
    # Handle inconsistent keys in dataset ('question' or 'query')
    query = props.get("question") or props.get("query", "")
    if not query:
        raise KeyError("No 'question' or 'query' key found in props")

    today_str = datetime.now().strftime("%Y-%m-%d")
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "critique": "",
        "iterations": 0,
        "today": today_str,
        "step": 1,
    }
    final_state = app.invoke(initial_state)
    final_answer_content = final_state["messages"][-1].content
    # The key 'answer' here is arbitrary but should match an 'outputs' key in the dataset.
    return {"output": {"answer": final_answer_content}}

# ==============================================================================
#
# CUSTOM EVALUATOR
#
# ==============================================================================
def correctness_evaluator(run, example):
    """A simple evaluator that checks if the expected answer is contained in the agent's output."""
    if run.outputs is None:
        return {"key": "correctness", "score": 0, "comment": "No output from run"}
    prediction_dict = run.outputs.get("output", {})
    if not isinstance(prediction_dict, dict):
        return {"key": "correctness", "score": 0, "comment": "Invalid output format"}
    prediction = prediction_dict.get("answer", "")
    reference = example.outputs.get("answer", "")
    score = 1 if reference.lower() in prediction.lower() else 0
    return {"key": "correctness", "score": score}

# ==============================================================================
#
# MAIN EXECUTION BLOCK
#
# ==============================================================================
def main():
    """Runs the automated evaluation process against an existing or new dataset."""
    print(colored("--- Starting LangSmith SDK Evaluation ---", "cyan"))

    # 1. Initialize the LangSmith client.
    client = Client()

    # 2. Define the name of the dataset.
    dataset_name = "ds-demo"

    # 3. Check if the dataset exists; create it from dataset.csv if not.
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(colored(f"Found existing dataset '{dataset_name}'.", "white"))
    except Exception as e:
        if "404" in str(e) or "not found" in str(e).lower():  # Dataset doesn't exist
            print(colored(f"Dataset '{dataset_name}' not found. Creating it programmatically...", "yellow"))
            # Load data from dataset.csv (assume it's in the same directory)
            import csv
            examples = []
            with open("dataset.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    examples.append({
                        "inputs": {"query": row["question"]},
                        "outputs": {"answer": row["answer"]}
                    })
            # Create the dataset and add examples
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="Synthetic QA dataset for agent evaluation"
            )
            client.create_examples(
                examples=examples,
                dataset_id=dataset.id
            )
            print(colored(f"Dataset '{dataset_name}' created successfully.", "green"))
        else:
            raise e  # Rethrow if it's not a "not found" error

    # 4. Run the Evaluation.
    print(colored("\nStarting evaluation run. This may take a few minutes...", "white"))
    
    try:
        experiment_results = evaluate(
            run_agent,
            data=dataset_name,
            experiment_prefix="Wikipedia-Agent-Correctness-SDK",
            evaluators=[correctness_evaluator],  # Use the custom callable evaluator
        )

        print(colored("\n--- Evaluation Complete ---", "cyan"))
        
        # Safely check for project_id instead of project_name
        if hasattr(experiment_results, 'project_id') and experiment_results.project_id:
            print(colored("View the detailed results in LangSmith:", "white"))
            project_url = f"https://smith.langchain.com/o/{client.tenant_handle}/projects/p/{experiment_results.project_id}"
            print(colored(project_url, "yellow"))
        else:
            print(colored("Evaluation finished, but no project link was generated.", "red"))
    except Exception as eval_error:
        print(colored(f"\nEvaluation failed: {eval_error}", "red"))
        print(colored("Check the partial results in the LangSmith UI using the link printed above.", "yellow"))

if __name__ == "__main__":
    main()
