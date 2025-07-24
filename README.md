# LangGraph Agent with LangSmith Evaluation

This project demonstrates the process of building a simple tool using agent with **LangGraph** and evaluating it using **LangSmith**. The goal is to showcase a realistic developer workflow: building a stateful LLM application, creating a test dataset, and running systematic evaluations to measure performance.

This repository is a response to the Sr TSE - LS Prompt.

### ✨ Features

  * **Stateful, Multi-step Agent**: The agent is built using LangGraph, allowing it to maintain conversational history in its state. It can perform multiple steps, like calling a model, then a tool, and then the model again, to arrive at a final answer.
  * **Dynamic Tool-Use with Tavily Search**: The agent is equipped with a search tool to find up-to-date information. It dynamically decides whether to use this tool based on the LLM's analysis of the user's query.
  * **Comprehensive Evaluation Suite**: The project includes multiple ways to test the agent, demonstrating a full development cycle:
      * **Offline Batch Testing**: A simple script for quick, local regression testing against a CSV file.
      * **Automated SDK Evaluation**: A script that programmatically runs the agent against a local CSV dataset and evaluates with LangSmith.
      * **Flexible Hybrid Evaluation**: A powerful script that uses command-line arguments and interactive prompts to test against different datasets and judge models configured in LangSmith, perfect for iterative testing.
  * **Custom LLM-as-a-Judge**: The evaluation scripts use a custom `CorrectnessEvaluator` class, which employs a separate LLM (like `gemini-2.0-flash`) to score the agent's answers for factual accuracy against a reference answer.

### 📂 Project Structure

```
.
├── agent.py                  # The core LangGraph agent definition.
├── dataset.csv                 # A synthetic dataset for evaluation.
├── evaluate.py                 # Simple script for offline batch testing against dataset.csv.
├── evaluate_langsmith.py       # Runs evaluation against dataset.csv using a hardcoded LLM judge.
├── evaluate_langsmith_custom.py# Runs evaluation against a specified LangSmith dataset and judge model.
├── setup.sh                    # Shell script to set up the environment.
├── .env                        # (You must create this) For API keys.
└── README.md                   # This file.
```

### 🚀 Setup and Installation

Follow these steps to set up your local environment.

1.  **Clone the Repository**
2.  **Create and Configure Environment Variables** in a `.env` file with your API keys.
3.  **Run the Setup Script**: `bash setup.sh`.
4.  **Activate the Virtual Environment**: `source venv/bin/activate`.

#### Example Setup Output

```
./setup.sh
Creating directory '/home/damian/Documents/Personal/Langchain/langgraph-agent'...
Creating new virtual environment in 'venv'...
...
Setup complete!
1. Remember to edit '/home/damian/Documents/Personal/Langchain/langgraph-agent/.env' with your actual API keys.
2. Activate your venv in new terminal sessions using: source /home/damian/Documents/Personal/Langchain/langgraph-agent/venv/bin/activate
```

### 🏗️ Agent Architecture (`agent.py`)

The core logic of the agent is defined in `agent.py` using LangGraph's state-machine paradigm. The architecture consists of a state, nodes, and conditional edges.

  * **State (`AgentState`)**: The agent's memory is a structured `TypedDict` that is passed between nodes and updated at each step.
  * **Nodes (`call_model`, `call_tool`)**: These are the processing units. `call_model` invokes the LLM, while `call_tool` executes the Tavily search.
  * **Conditional Edges (`should_continue`)**: This is the logic that connects the nodes, creating a loop. It checks if the model wants to call a tool and routes the flow accordingly.

### 💬 Interacting with the Agent

While the evaluation scripts focus on batch processing, the core agent from `agent.py` can be used for direct, single-question interactions. This is the best way to perform a quick, qualitative check of the agent's abilities.

#### Example Interaction

```
$ python3 agent.py
Welcome to the Wikipedia-powered AI Agent!
Type your question (or 'exit' to quit):
Your question: How large is the United States of America?

--- Final Answer ---
The total area of the United States is 3,796,742 square miles (9,833,520 km2). The land area is 3,531,905 square miles (9,147,590 km2).
```

### ⚙️ Automated Evaluation Workflows

This project provides three distinct methods for running automated, batch evaluations.

#### Method 1: Offline Batch Evaluation (`evaluate.py`)

This is the most basic evaluation method. It runs the agent against the local `dataset.csv` file to perform a quick, static regression test without any dependency on LangSmith for scoring.

  * **Purpose**: Quick, offline validation to ensure the agent runs without errors across a set of questions.
  * **Usage**: `python3 evaluate.py`
  * **Example Output**:
    ```
    --- Starting Agent Evaluation ---

    [1/8] Running test for query: 'What is the capital of Japan?'
      - Agent's Answer: The capital of Japan is Tokyo.
      [PASS]
    ...
    --- Evaluation Summary ---
    Total Tests: 8
      Passed: 8
      Failed: 0
    ```

#### Method 2: Automated Evaluation with LangSmith (`evaluate_langsmith.py`)

This script elevates the testing by integrating LangSmith for automated tracing and scoring, while still using the local dataset file.

  * **Purpose**: To run a reproducible evaluation against a local dataset with scoring done by an LLM-as-a-judge.
  * **Usage**: `python3 evaluate_langsmith.py`
  * **Example Output**:
    ```
    --- Starting LangSmith SDK Evaluation ---
    Found existing dataset 'ds-demo'.

    Starting evaluation run. This may take a few minutes...
    View the evaluation results for experiment: 'Wikipedia-Agent-Correctness-SDK-4da73057' at:
    https://smith.langchain.com/o/4acdec47-e18b-4d6f-a676-6a4312603733/datasets/ea56c294-7249-4a23-84ab-dfcb626c9afd/compare?selectedSessions=375f7299-69fa-48d2-a01b-fb849d06b131

    --- Evaluation Complete ---
    ```

#### Method 3: Hybrid Evaluation with UI + SDK (`evaluate_langsmith_custom.py`)

This is the most powerful evaluation method, demonstrating the synergy between the LangSmith UI for data management and the SDK for execution.

  * **Purpose**: To run experiments against centrally-managed datasets in the LangSmith UI, with the flexibility to change parameters via the command line or interactive prompts.
  * **Usage**:
      * Interactive: `python3 evaluate_langsmith_custom.py --debug`
      * Non-Interactive: `python3 evaluate_langsmith_custom.py --dataset ds-loyal-ceramics-83 --model gemini-2.0-flash`

### 🤔 Learnings & Friction Log

While building this project, I noted several hurdles and learnings that informed the final design. This log addresses the "what might confuse a new user" and "what you learned" aspects of the prompt.

  * **UI-Driven Evaluation Workflow**: A key point of friction is the user journey for initiating evaluations on custom agents. After creating a dataset in the UI, a user's natural next step is to look for a "Run" button to test their local agent, but the primary path requires switching to the SDK.
  * **Evaluator Precedence (UI vs. SDK)**: The relationship between evaluators defined in the UI versus those defined in the SDK can be confusing. It's not immediately obvious that an evaluator defined in a Python script will silently override a UI-based evaluator with the same feedback key during an SDK-driven run.
  * **Local Model Performance**: Initial attempts to use local Small Language Models (SLMs) like Microsoft's Phi resulted in slow response times (minutes per query) on CPU-only hardware. This necessitated a pivot to a more performant, cloud-based API like Google's Gemini to ensure a reasonable user experience.
  * **Advanced Tooling for Retrieval**: A simple, direct Wikipedia query tool was initially used but proved insufficient for complex questions. The agent's accuracy was significantly improved by pivoting to a more robust, two-step retrieval process: first using `wikipedia_url_searcher` to find a relevant page, and then using `web_page_reader` to extract the specific information.
  * **Model-Specific Parsing**: Integrating with the Gemini API introduced some unique parsing requirements within LangGraph, particularly around how tool calls are structured in the model's output. This highlighted that agent logic can be tightly coupled to the specific model provider's output format, and may require different handling compared to models from OpenAI.
  * **Guardrails Integration Complexity**: An early development goal was to include Guardrails for response moderation. However, integrating this layer with the existing LangGraph state machine proved complex and was ultimately removed to focus on the core agent and evaluation workflow.
