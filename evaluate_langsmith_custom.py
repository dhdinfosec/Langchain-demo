# ==============================================================================
#
# LangSmith SDK Evaluation Script with Custom LLM-as-a-Judge Evaluator (Gemini)
#
# This script evaluates a LangGraph agent defined in 'agent.py' against a 
# dataset (configurable via --dataset) using the LangSmith SDK. It runs the agent on each 
# example in the dataset, generates predictions, and scores them using a 
# custom evaluator powered by Gemini API as a judge for correctness.
#
# Key Features:
# - Fetches the dataset from LangSmith.
# - Invokes the agent for each example.
# - Uses a custom evaluator to judge if the agent's answer matches the 
#   reference (expected) answer.
# - Handles errors gracefully (e.g., KeyErrors in props, Gemini API issues).
# - Outputs progress and results to the console in a readable format.
# - Supports --debug flag for verbose logging; otherwise, shows clean progress.
# - Supports --dataset to specify the dataset name (default: ds-demo).
# - Supports --model to specify the Gemini model name (default: gemini-1.5-flash).
#
# Usage:
# - Run normally: python evaluate_langsmith_custom.py (will prompt for parameters)
# - With debug: python evaluate_langsmith_custom.py --debug
# - With custom dataset: python evaluate_langsmith_custom.py --dataset ds-loyal-ceramics-83
# - With custom model: python evaluate_langsmith_custom.py --model gemini-1.0-flash
# - Combined: python evaluate_langsmith_custom.py --dataset ds-loyal-ceramics-83 --model gemini-1.0-flash
#
# Requirements:
# - .env file with LANGSMITH_API_KEY and GOOGLE_API_KEY.
# - 'agent.py' in the same directory with the compiled LangGraph app.
# - Dataset 'ds-demo' (or specified dataset) imported into LangSmith.
#
# ==============================================================================

# --- Core Imports ---
import os
import sys
import logging
import argparse
from datetime import datetime
from typing import Dict
from uuid import UUID

# --- Third-party Library Imports ---
from dotenv import load_dotenv
load_dotenv(override=True)
from langsmith import Client
from langsmith.evaluation import evaluate, run_evaluator
from langsmith.schemas import Feedback
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs): return text
    logging.warning("'termcolor' not found; install for colored output.")

# --- Local Application Imports ---
import agent
from agent import app, AgentState

# ==============================================================================
# SCRIPT CONFIGURATION AND ARGUMENT PARSING
# ==============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Evaluate LangGraph Agent with LangSmith.")
parser.add_argument('--debug', action='store_true', help='Enable debug mode for verbose output.')
# Store defaults so we can check if they were provided via command line later
parser.add_argument('--dataset', type=str, default='_DEFAULT_DATASET_PLACEHOLDER_',
                    help='Name of the dataset in LangSmith to use for evaluation (default: ds-demo).')
parser.add_argument('--model', type=str, default='_DEFAULT_MODEL_PLACEHOLDER_',
                    help='Gemini model to use for the LLM-as-a-Judge evaluator (default: gemini-1.5-flash).')
args = parser.parse_args()

debug_mode = args.debug
agent.debug_mode = debug_mode

if debug_mode:
    logger.setLevel(logging.DEBUG)
    print(colored("Debug mode is ON.", "yellow"))

load_dotenv(override=True)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LangGraph Wikipedia Agent - Evaluation")

if not os.getenv("LANGSMITH_API_KEY"):
    logger.error("LANGSMITH_API_KEY not set. Please set it in your .env file or environment variables.")
    sys.exit(1)
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("GOOGLE_API_KEY not set. Please set it in your .env file or environment variables.")
    sys.exit(1)

# --- Parameter Prompting Logic ---
dataset_name = args.dataset
evaluator_model_name = args.model

# Prompt for dataset name if not provided via command line
if dataset_name == '_DEFAULT_DATASET_PLACEHOLDER_':
    prompted_dataset = input(colored("Enter the LangSmith dataset name (default: ds-demo): ", "yellow")).strip()
    if not prompted_dataset:
        print(colored("No dataset name provided. Exiting gracefully.", "red"))
        sys.exit(0)
    dataset_name = prompted_dataset
else:
    # If provided via command line, ensure it's not an empty string
    if not dataset_name:
        logger.error("Dataset name cannot be an empty string. Exiting.")
        sys.exit(1)

# Prompt for model name if not provided via command line
if evaluator_model_name == '_DEFAULT_MODEL_PLACEHOLDER_':
    prompted_model = input(colored("Enter the Gemini model name for evaluation (default: gemini-1.5-flash): ", "yellow")).strip()
    if not prompted_model:
        print(colored("No model name provided. Exiting gracefully.", "red"))
        sys.exit(0)
    evaluator_model_name = prompted_model
else:
    # If provided via command line, ensure it's not an empty string
    if not evaluator_model_name:
        logger.error("Model name cannot be an empty string. Exiting.")
        sys.exit(1)


google_api_key = os.getenv("GOOGLE_API_KEY")

logger.info(f"Evaluation will use dataset: '{dataset_name}'")
logger.info(f"LLM-as-a-Judge will use model: '{evaluator_model_name}'")

# ==============================================================================
# AGENT INVOCATION LOGIC
# ==============================================================================
def run_agent(props: dict):
    """The target function for the evaluation, running the agent for a single data point."""
    # Handle inconsistent keys in dataset ('question' or 'query') for robustness
    query = props.get("question") or props.get("query", "")
    if not query:
        raise KeyError("No 'question' or 'query' key found in dataset example inputs.")

    today_str = datetime.now().strftime("%Y-%m-%d")

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)], "critique": "", "iterations": 0, "today": today_str, "step": 1,
    }

    final_state = app.invoke(initial_state)
    final_answer = final_state["messages"][-1]
    
    # LangSmith evaluators expect a dictionary with 'output' key for the prediction.
    # The content of the 'output' key should be the agent's final answer.
    return {"output": final_answer.content}

# ==============================================================================
# CUSTOM LLM-AS-A-JUDGE EVALUATOR
# ==============================================================================

evaluator_llm = ChatGoogleGenerativeAI(model=evaluator_model_name, temperature=0, google_api_key=google_api_key)

# Changed from "system" to "human" to ensure proper content delivery for Gemini models.
# Gemini models often expect prompt content within a HumanMessage for generation.
evaluator_prompt = ChatPromptTemplate.from_messages([
    ("human", """You are an impartial judge. Your task is to assess whether the 'predicted_answer' correctly and comprehensively answers the 'question' based on the 'reference_answer'.
    
    If the 'predicted_answer' is correct, complete, and aligns with the 'reference_answer', respond with '[[1]]' and a brief comment.
    If the 'predicted_answer' is incorrect, incomplete, or contradicts the 'reference_answer', respond with '[[0]]' and a brief comment explaining why.
    
    Question: {question}
    Reference Answer: {reference_answer}
    Predicted Answer: {predicted_answer}
    
    Your Response:
    """),
])

@run_evaluator
def correctness_evaluator(run, example):
    """Custom LLM-as-a-Judge evaluator for correctness."""
    # Extract inputs from the LangSmith 'run' and 'example' objects
    question_text = example.inputs.get("query") or example.inputs.get("question")
    reference_answer_text = example.outputs["answer"]
    predicted_answer_text = run.outputs["output"]

    # Explicitly format the prompt into messages that the ChatGoogleGenerativeAI model expects
    messages_for_evaluator = evaluator_prompt.format_messages(
        question=question_text,
        reference_answer=reference_answer_text,
        predicted_answer=predicted_answer_text
    )

    # Invoke the LLM directly with the prepared list of messages
    # This bypasses a potential issue where chaining ChatPromptTemplate directly to
    # ChatGoogleGenerativeAI might not correctly convert input types in some versions.
    llm_output = evaluator_llm.invoke(messages_for_evaluator)

    # Extract the content from the LLM's BaseMessage output
    eval_output = llm_output.content

    # Parse the LLM's output to extract the score and comment
    score = 1 if "[[1]]" in eval_output else 0
    comment = eval_output.replace("[[1]]", "").replace("[[0]]", "").strip()

    # Return a dictionary. The @run_evaluator decorator will convert this
    # into a Feedback object and populate the required metadata fields (id, created_at, modified_at).
    return {
        "key": "Correctness (Gemini Judge)", # This key will appear in LangSmith UI
        "score": score,
        "comment": comment,
        "source_run_id": run.id,
        "feedback_source_type": "MODEL", # Indicates that the feedback came from a model
    }

# ==============================================================================
# MAIN EVALUATION EXECUTION
# ==============================================================================

def main():
    client = Client()

    logger.info(f"Attempting to fetch dataset '{dataset_name}' from LangSmith...")
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        logger.info(f"Dataset '{dataset_name}' fetched successfully (ID: {dataset.id}).")
    except Exception as e:
        logger.error(f"Failed to fetch dataset '{dataset_name}': {e}")
        print(colored(f"\nError: Could not find dataset '{dataset_name}' in LangSmith.", "red"))
        print(colored("Please ensure the dataset exists and your LANGCHAIN_API_KEY has access.", "yellow"))
        sys.exit(1)

    print(colored("\nStarting evaluation run. This may take a few minutes...", "white"))
    
    try:
        experiment_results = evaluate(
            run_agent,
            data=dataset.id, 
            evaluators=[correctness_evaluator], # Pass the decorated evaluator function
            experiment_prefix="Wikipedia-Agent-Correctness-SDK",
        )

        print(colored("\n--- Evaluation Complete ---", "cyan", attrs=["bold"]))
        
        print(colored(f"Total Runs: {len(experiment_results.runs)}", "white"))
        
        total_score = 0
        for run_id, result in experiment_results.results.items():
            if result.get("feedback"):
                for feedback_item in result["feedback"]:
                    if feedback_item.get("score") == 1:
                        total_score += 1
        
        if experiment_results.runs:
            pass_rate = (total_score / len(experiment_results.runs)) * 100
            print(colored(f"Overall Correctness Score: {total_score}/{len(experiment_results.runs)} ({pass_rate:.2f}%)", "white", attrs=["bold"]))

        print(colored("\n--- Detailed Results ---", "yellow"))
        for i, (run_id, result) in enumerate(experiment_results.results.items()):
            example = next((ex for ex in dataset.examples if ex.id == result['example_id']), None)
            question = (example.inputs.get("question") or example.inputs.get("query")) if example else "N/A"
            reference = example.outputs.get("answer") if example else "N/A"
            predicted = result['outputs'].get('output') if result.get('outputs') else "No Output"

            score = "N/A"
            comment = "No comment"
            if result.get("feedback"):
                for feedback_item in result["feedback"]:
                    score = feedback_item.get("score", "N/A")
                    comment = feedback_item.get("comment", "No comment")
                    break 

            print(f"{i+1}. Question: {question}")
            print(f"   Reference: {reference}")
            print(f"   Predicted: {predicted}")
            score_color = "green" if score == 1 else ("red" if score == 0 else "white")
            print(colored(f"   Score: {score} ({comment})", score_color))
            print("---")
        
        project_id = getattr(experiment_results, 'project_id', None)
        if project_id:
            print(colored("View the detailed results in LangSmith:", "white", attrs=["bold"]))
            tenant_handle = client.tenant_handle if hasattr(client, 'tenant_handle') else 'default'
            project_url = f"https://smith.langchain.com/o/{tenant_handle}/projects/p/{project_id}"
            print(colored(project_url, "yellow", attrs=["underline"]))
            logger.info(f"Evaluation logged to project ID: {project_id}")
        else:
            print(colored("Evaluation finished, but no project link was generated. Check your LangSmith API key and project settings.", "red", attrs=["bold"]))
            logger.warning("No project_id found in experiment_results.")

    except Exception as eval_error:
        logger.error(f"Evaluation failed: {eval_error}", exc_info=True)
        print(colored(f"\nEvaluation failed unexpectedly: {eval_error}", "red", attrs=["bold"]))
        print(colored("Please check the debug log for details or ensure your LangSmith/Google API keys are correct and the dataset exists.", "yellow"))


if __name__ == "__main__":
    main()
