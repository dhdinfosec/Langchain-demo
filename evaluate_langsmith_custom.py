# ==============================================================================
#
# LangSmith SDK Evaluation Script with Custom LLM-as-a-Judge Evaluator (Gemini)
#
# This script evaluates a LangGraph agent defined in 'agent.py' against a 
# dataset ('ds-demo') using the LangSmith SDK. It runs the agent on each 
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
#
# Usage:
# - Run normally: python evaluate_langsmith_custom.py
# - With debug: python evaluate_langsmith_custom.py --debug
#
# Requirements:
# - .env file with LANGSMITH_API_KEY and GOOGLE_API_KEY.
# - 'agent.py' in the same directory with the compiled LangGraph app.
# - Dataset 'ds-demo' imported into LangSmith.
#
# ==============================================================================

# --- Core Imports ---
import os
import sys  # Added for command-line argument parsing
import logging
from datetime import datetime
from typing import Dict
from uuid import UUID  # For UUID comparison in project verification

# Configure logging (level set later based on --debug)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Default to INFO
logger = logging.getLogger(__name__)

# --- Third-party Library Imports ---
from dotenv import load_dotenv
load_dotenv(override=True)  # Override any cached env vars for fresh load
from langsmith import Client, evaluate
from langsmith.evaluation import RunEvaluator, EvaluationResult
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from termcolor import colored

# --- Local Application Imports ---
from agent import app, AgentState

# ==============================================================================
#
# SCRIPT CONFIGURATION
#
# ==============================================================================
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_TRACING_LOG_LEVEL"] = "debug"  # For verbose LangSmith traces if --debug is enabled
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LangGraph Wikipedia Agent - Custom Eval")
EXPECTED_PROJECT_ID = UUID("225e2bbf-7add-4f5d-b49a-5176ab971c8e")  # Your project ID as UUID for verification
TENANT_HANDLE = "4acdec47-e18b-4d6f-a676-6a4312603733"  # Your tenant/org handle from logs/UI
google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    raise RuntimeError("GOOGLE_API_KEY not set.")

# Initialize Gemini model for the evaluator (used to judge correctness)
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, google_api_key=google_key)

# Check for --debug flag and set logging level accordingly
debug_mode = '--debug' in sys.argv
if debug_mode:
    logging.getLogger().setLevel(logging.DEBUG)  # Enable verbose debug logs
    logger.debug("Debug mode enabled.")

# ==============================================================================
#
# AGENT INVOCATION LOGIC (Target for Evaluation)
#
# This function wraps the invocation of the LangGraph agent from 'agent.py'.
# It extracts the query from the dataset props, runs the agent, and returns
# the output in the format expected by LangSmith ({"output": {"answer": ...}}).
# Handles errors by logging and returning error metadata.
#
# ==============================================================================
def run_agent(props: dict) -> dict:
    """Runs the agent from agent.py for a single data point."""
    logger.debug(f"run_agent called with props: {props}")
    try:
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
        output = {"output": {"answer": final_answer_content}}
        logger.debug(f"Agent output: {output}")
        return output
    except KeyError as ke:
        logger.error(f"KeyError in run_agent: {ke} - Props structure: {props}")
        return {"output": {"answer": "", "error": f"KeyError: {ke}"}}
    except Exception as e:
        logger.error(f"Exception in run_agent: {e} - Props: {props}")
        return {"output": {"answer": "", "error": str(e)}}

# ==============================================================================
#
# CUSTOM LLM-AS-A-JUDGE EVALUATOR (Using Gemini)
#
# This class defines a custom evaluator for LangSmith. It uses Gemini to judge
# if the agent's predicted answer is correct relative to the reference answer.
# - Scores 1 (correct) or 0 (incorrect).
# - Handles errors (e.g., API failures, parsing issues).
# - Strips markdown from Gemini responses for reliable JSON parsing.
#
# ==============================================================================
class AnswerCorrectnessEvaluator(RunEvaluator):
    """Custom evaluator using Gemini as a judge to check answer correctness."""

    def evaluate_run(self, run, example, evaluator_run_id=None) -> EvaluationResult:  # Added evaluator_run_id=None to handle unexpected kwargs
        logger.debug(f"Evaluating run: {run.id} - Example inputs: {example.inputs}, outputs: {example.outputs}")
        if run.outputs is None:
            logger.warning(f"No outputs for run {run.id}")
            return EvaluationResult(key="correctness", score=0, comment="No output from run")

        predicted_answer = run.outputs.get("output", {}).get("answer", "")
        error = run.outputs.get("output", {}).get("error", "")
        if error:
            logger.warning(f"Run {run.id} has error: {error}")
            return EvaluationResult(key="correctness", score=0, comment=f"Run failed: {error}")

        reference_answer = example.outputs.get("answer", "")
        question = example.inputs.get("query", "") or example.inputs.get("question", "")

        instructions = """\
Given the following question, reference answer, and predicted answer, determine if the predicted answer is correct, accurate, and consistent with the reference answer.

Output a JSON object with a single key: "is_correct" (boolean: true if correct, false otherwise).
"""

        msg = f"Question: {question}\nReference Answer: {reference_answer}\nPredicted Answer: {predicted_answer}"
        prompt = instructions + "\n" + msg

        if not question or not reference_answer or not predicted_answer:
            logger.warning(f"Incomplete data for evaluation in run {run.id}: question={bool(question)}, reference={bool(reference_answer)}, predicted={bool(predicted_answer)}")
            return EvaluationResult(key="correctness", score=0, comment="Incomplete data for evaluation")

        # Call Gemini
        try:
            response = gemini_model.invoke(prompt)
            response_text = response.content.strip()
            logger.debug(f"Gemini response for run {run.id}: {response_text}")
        except Exception as e:
            logger.error(f"Gemini API error in evaluator for run {run.id}: {e}")
            return EvaluationResult(key="correctness", score=0, comment=f"Gemini API error: {e}")

        # Strip markdown from response for reliable parsing
        response_text = response_text.strip().removeprefix('```json').removesuffix('```').strip()

        # Parse the response
        try:
            import json
            parsed = json.loads(response_text)
            is_correct = parsed.get("is_correct", False)
            score = 1 if is_correct else 0
            comment = "Correct" if is_correct else "Incorrect"
        except Exception as e:
            logger.error(f"Parse error in evaluator for run {run.id}: {e} - Response: {response_text}")
            score = 0
            comment = f"Failed to parse Gemini response: {e}"

        return EvaluationResult(key="correctness", score=score, comment=comment)

# ==============================================================================
#
# MAIN EXECUTION BLOCK
#
# This function orchestrates the entire evaluation:
# - Verifies the project in LangSmith.
# - Loads the dataset.
# - Runs the evaluation using LangSmith's evaluate() function.
# - Prints progress for each example (question, agent's answer, judge's score).
# - Handles the experiment results, logging details if --debug is enabled.
#
# ==============================================================================
def main():
    """Runs the evaluation with the custom Gemini judge."""
    print(colored("--- Starting LangSmith SDK Custom Evaluation ---", "cyan"))

    # Initialize the LangSmith client.
    client = Client()

    # Verify project
    try:
        projects = list(client.list_projects())
        project = next((p for p in projects if p.id == EXPECTED_PROJECT_ID), None)
        if project:
            logger.info(f"Found project 'LangGraph RAG Agent' with ID {EXPECTED_PROJECT_ID}")
        else:
            logger.warning(f"Project ID {EXPECTED_PROJECT_ID} not found in projects: {[p.id for p in projects]}")
            print(colored("Warning: Expected project not found. Results may log to a different project.", "yellow"))
    except Exception as e:
        logger.error(f"Failed to list projects: {e}")

    # Define the dataset name (use existing 'ds-demo').
    dataset_name = "ds-demo"

    # Fetch the dataset.
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(colored(f"Found existing dataset '{dataset_name}'.", "white"))
        logger.info(f"Dataset ID: {dataset.id}")
    except Exception as e:
        print(colored(f"Error fetching dataset: {e}", "red"))
        logger.error(f"Dataset fetch error: {e}")
        sys.exit(1)

    # Run the Evaluation with the custom evaluator.
    print(colored("\nStarting evaluation run. This may take a few minutes...", "white"))
    
    try:
        experiment_results = evaluate(
            run_agent,
            data=dataset_name,
            experiment_prefix="Wikipedia-Agent-Custom-Gemini-Judge",
            evaluators=[AnswerCorrectnessEvaluator()],  # Use the custom Gemini evaluator
        )
        logger.debug(f"Experiment results attributes: {dir(experiment_results)}")
        logger.debug(f"Experiment results vars: {vars(experiment_results) if hasattr(experiment_results, '__dict__') else 'No vars available'}")

        print(colored("\n--- Evaluation Complete ---", "cyan"))
        
        # Print readable summary of results
        print("\nEvaluation Results Summary:")
        for i, result in enumerate(experiment_results, 1):
            run = result['run']
            example = result['example']
            eval_results = result['evaluation_results']['results']
            eval_result = eval_results[0] if eval_results else None
            question = example.inputs.get("question") or example.inputs.get("query", "Unknown question")
            predicted = run.outputs.get("output", {}).get("answer", "No answer")
            reference = example.outputs.get("answer", "No reference")
            
            if eval_result:
                if isinstance(eval_result, dict):
                    score = eval_result.get("score", 0)
                    comment = eval_result.get("comment", "No comment")
                else:
                    score = getattr(eval_result, 'score', 0)
                    comment = getattr(eval_result, 'comment', "No comment")
            else:
                score = 0
                comment = "No evaluation"

            print(f"{i}. Question: {question}")
            print(f"   Reference: {reference}")
            print(f"   Predicted: {predicted}")
            print(colored(f"   Score: {score} ({comment})", "green" if score == 1 else "red"))
            print("---")
        
        # Check for project_id or other attributes (use EXPECTED_PROJECT_ID as fallback if verified)
        project_id = getattr(experiment_results, 'project_id', None) or getattr(experiment_results, 'experiment_id', None)
        if project_id is None:
            project_id = EXPECTED_PROJECT_ID  # Use verified project ID to avoid warning
            logger.info(f"Using verified project ID {project_id} as fallback for URL.")
        if project_id:
            print(colored("View the detailed results in LangSmith:", "white"))
            project_url = f"https://smith.langchain.com/o/{TENANT_HANDLE}/projects/p/{project_id}"
            print(colored(project_url, "yellow"))
            logger.info(f"Evaluation logged to project ID: {project_id}")
        else:
            print(colored("Evaluation finished, but no project link was generated.", "red"))
            logger.warning("No project_id or experiment_id found in experiment_results")
    except Exception as eval_error:
        logger.error(f"Evaluation failed: {eval_error}")
        print(colored(f"\nEvaluation failed: {eval_error}", "red"))
        print(colored("Check partial results in the LangSmith UI.", "yellow"))

if __name__ == "__main__":
    main()
