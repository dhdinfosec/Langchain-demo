# LangGraph "Search-then-Read" RAG Agent (Wikipedia-Only Edition)

This repository contains a simple, yet sophisticated, conversational AI agent built using LangGraph. The agent mimics a human research process to answer questions by exclusively utilizing Wikipedia as its external information source. This project also demonstrates how to set up an evaluation pipeline using a local Python script for basic testing.

## Features

* **Constrained Two-Step RAG ("Search-then-Read")**: The agent employs a specific two-step process:
    * `wikipedia_url_searcher`: A tool (powered by Tavily) that is hard-coded to search only within `en.wikipedia.org` for relevant URLs.
    * `web_page_reader`: A tool that reads the full content of a URL provided by the searcher.
* **Advanced Agentic Reasoning**: The agent's core prompt guides it through this specific research process, allowing it to answer from its own knowledge for common questions or resort to tools for more complex or current information.
* **CriticNode**: A final quality gate that evaluates the agent's synthesized answer and requests revisions if the response is unsatisfactory.
* **Local Evaluation Script**: A Python script (`evaluate.py`) is provided to run a predefined suite of tests against the agent and report pass/fail results based on expected keywords in the answers.
* **LangSmith SDK Evaluation Script**: A Python script (`evaluate_langsmith_custom.py`) is provided to run comprehensive evaluations against the agent using the LangSmith SDK, including a custom LLM-as-a-Judge for scoring.

## Project Structure

* `setup.sh`: A bash script to automate the project environment setup, including virtual environment creation and package installation.
* `agent.py`: Contains the core LangGraph agent implementation, defining its state, nodes (Agent, Tool Executor, Critic), and the logic (edges) that connect them.
* `evaluate.py`: A Python script for running automated local evaluations of the agent using a keyword-based assertion approach.
* `evaluate_langsmith_custom.py`: A Python script for running automated evaluations of the agent using the LangSmith SDK and a custom LLM-as-a-Judge.
* `README.md` (this file): Provides an overview of the project, setup instructions, and how to run the agent and its evaluations.

## Getting Started

Follow these steps to set up the project and run the agent and its evaluations.

### Prerequisites

* Python 3.9+
* `bash` (for running the setup script)

### 1. Setup the Environment

The `setup.sh` script automates the creation of a virtual environment and installation of necessary Python packages.

1.  **Run the setup script:**
    ```bash
    bash setup.sh
    ```
    This script will:
    * Create a `langgraph-agent` directory in your current location if it doesn't exist.
    * Create and activate a Python virtual environment named `venv` inside `langgraph-agent`.
    * Install all required Python packages (e.g., `langgraph`, `langchain-google-genai`, `langsmith`, `python-dotenv`, `langchain-tavily`, etc.).
    * Create a `.env` file in the `langgraph-agent` directory with placeholders for your API keys.

2.  **Important: Configure your API Keys:**
    After running `setup.sh`, you **must** edit the newly created `.env` file within the `langgraph-agent` directory. Replace the placeholder values with your actual API keys:

    ```ini
    LANGSMITH_API_KEY="your-langsmith-key-here"
    GOOGLE_API_KEY="your-gemini-api-key-here"
    TAVILY_API_KEY="your-tavily-api-key-here"
    ```
    * **`LANGSMITH_API_KEY`**: Required for LangSmith evaluation.
    * **`GOOGLE_API_KEY`**: Required for the `ChatGoogleGenerativeAI` models used by the agent and the custom LLM-as-a-Judge.
    * **`TAVILY_API_KEY`**: Required for the `wikipedia_url_searcher` tool.

### 2. Run the Agent (Interactive Mode)

You can interact with the agent directly from your terminal.

1.  **Ensure your virtual environment is active.** If you open a new terminal, navigate to the `langgraph-agent` directory and activate the environment:
    ```bash
    cd langgraph-agent
    source venv/bin/activate
    ```
2.  **Run the agent:**
    ```bash
    python3 agent.py
    ```
3.  Type your questions at the prompt (e.g., "What is the capital of France?") and press Enter. Type `exit` to quit.

    By default, the agent runs in a concise mode:
    ```
    Welcome to the Wikipedia-powered AI Agent!
    Type your question (or 'exit' to quit):
    Your question: Who wrote Hamlet?

    --- Final Answer ---
    William Shakespeare wrote Hamlet.
    ```

    You can also run the agent with a `--debug` parameter to see more verbose output about its internal steps. The `--debug` parameter provides detailed insights into the agent's thought process, including each step of its reasoning, tool execution, and the critic's evaluation.

    With `--debug`:
    ```bash
    python3 agent.py --debug
    ```
    Example output with `--debug`:
    ```
    Welcome to the Wikipedia-powered AI Agent!
    Type your question (or 'exit' to quit):
    Your question: What is the population of England?

    --- AGENT ITERATION 1 ---
      Step 1: Planning next action...
      Router: Analyzing agent's last message (type: AIMessage).
      Router: Decision -> Execute tools.
      Step 2: Executing tool 'wikipedia_url_searcher'...
      Step 3: Deciding which page to read...
      Router: Analyzing agent's last message (type: AIMessage).
      Router: Decision -> Execute tools.
      Step 4: Executing tool 'web_page_reader'...
      Step 5: Synthesizing final answer...
      Router: Analyzing agent's last message (type: AIMessage).
      Router: Decision -> Go to critic.
      Step 6: Evaluating final answer...
    Critic response: ACCEPT

    --- Final Answer ---
    The population of England was estimated to be 57,106,398 in 2022. The 2021 census recorded the population as 56,490,048.
    ```

### 3. Run the Automated Local Evaluation

The `evaluate.py` script allows you to test the agent against a predefined dataset using a local keyword-based assertion approach.

1.  **Ensure your virtual environment is active** (see step 1 above).
2.  **Run the evaluation script:**
    ```bash
    python3 evaluate.py
    ```
    or, to see verbose debugging output from the agent during evaluation:
    ```bash
    python3 evaluate.py --debug
    ```
3.  The script will iterate through a set of predefined queries, send them to the agent, and check if the agent's answer contains specific keywords. A summary report will be printed at the end, indicating how many tests passed and failed.

    *Example Output Snippet:*
    ```
    --- Starting Agent Evaluation ---

    [1/8] Running test for query: 'What is the capital of Japan?'
      - Agent's Answer: The capital of Japan is Tokyo.
      [PASS]

    [2/8] Running test for query: 'Who invented the telephone?'
      - Agent's Answer: The telephone was invented by Alexander Graham Bell.
      [PASS]

    ...

    --- Evaluation Summary ---
    Total Tests: 8
      Passed: 8
      Failed: 0
    ```

### 4. Run the Automated LangSmith Evaluation

The `evaluate_langsmith_custom.py` script allows you to run a comprehensive evaluation of the agent using the LangSmith SDK. This script utilizes a custom LLM-as-a-Judge evaluator (powered by Gemini) to score the agent's answers against a reference dataset.

1.  **Ensure your virtual environment is active** (see step 1 above).
2.  **Ensure you have created a dataset named `ds-demo` in LangSmith** with the questions and reference answers you wish to evaluate against.
3.  **Run the evaluation script:**
    ```bash
    python3 evaluate_langsmith_custom.py
    ```
    You can also run with a `--debug` parameter for more verbose logging during the evaluation process:
    ```bash
    python3 evaluate_langsmith_custom.py --debug
    ```
4.  The script will fetch the `ds-demo` dataset from LangSmith, invoke the agent for each example, and use the custom Gemini-powered judge to score the answers. It will then print a summary of the results to the console, including the score for each question.

    *Example Output:*
    ```
    --- Starting LangSmith SDK Custom Evaluation ---
    2025-07-23 03:08:02,047 - INFO - Found project 'LangGraph RAG Agent' with ID 225e2bbf-7add-4f5d-b49a-5176ab971c8e
    Found existing dataset 'ds-demo'.
    2025-07-23 03:08:03,006 - INFO - Dataset ID: ea56c294-7249-4a23-84ab-dfcb626c9afd

    Starting evaluation run. This may take a few minutes...
    View the evaluation results for experiment: 'Wikipedia-Agent-Custom-Gemini-Judge-2a9579fa' at:
    [https://smith.langchain.com/o/4acdec47-e18b-4d6f-a676-6a4312603733/datasets/ea56c294-7249-4a23-84ab-dfcb626c9afd/compare?selectedSessions=a9e31572-b251-46c0-ac67-56802695e742](https://smith.langchain.com/o/4acdec47-e18b-4d6f-a676-6a4312603733/datasets/ea56c294-7249-4a23-84ab-dfcb626c9afd/compare?selectedSessions=a9e31572-b251-46c0-ac67-56802695e742)


    8it [00:45,  5.68s/it]

    --- Evaluation Complete ---

    Evaluation Results Summary:
    1. Question: Who wrote Hamlet?
       Reference: Hamlet was written by William Shakespeare.
       Predicted: William Shakespeare wrote Hamlet.
       Score: 1 (Correct)
    ---
    2. Question: Who invented the telephone?
       Reference: The telephone was invented by Alexander Graham Bell.
       Predicted: Alexander Graham Bell is credited with inventing the telephone.
       Score: 1 (Correct)
    ---
    3. Question: What is the population of Canada?
       Reference: The population of Canada is over 38 million people.
       Predicted: As of 2024, Canada's population is estimated to be over 41.5 million.
       Score: 1 (Correct)
    ---
    4. Question: What is the capital of Japan?
       Reference: The capital of Japan is Tokyo.
       Predicted: The capital of Japan is Tokyo.
       Score: 1 (Correct)
    ---
    5. Question: When was the Eiffel Tower built?
       Reference: The Eiffel Tower was completed in 1889.
       Predicted: The Eiffel Tower was built from 1887 to 1889. Construction started on 28 January 1887, and it was completed on 31 March 1889. The tower was opened on 15 May 1889.
       Score: 1 (Correct)
    ---
    6. Question: What is the chemical symbol for gold?
       Reference: The chemical symbol for gold is Au.
       Predicted: The chemical symbol for gold is Au.
       Score: 1 (Correct)
    ---
    7. Question: What is the tallest mountain in the world?
       Reference: The tallest mountain in the world is Mount Everest.
       Predicted: The tallest mountain in the world is Mount Everest.
       Score: 1 (Correct)
    ---
    8. Question: What is Java? (programming language)
       Reference: Java is a high-level, class-based, object-oriented programming language.
       Predicted: Java is a high-level, general-purpose, memory-safe, object-oriented programming language. It was designed to allow programmers to "write once, run anywhere" (WORA), meaning that compiled Java code can run on any platform that supports Java without recompilation. Java applications are typically compiled to bytecode that can run on any Java virtual machine (JVM). The syntax of Java is similar to C and C++, but it has fewer low-level facilities.

    Java was designed by James Gosling at Sun Microsystems and released in May 1995. It has gained popularity since its release and has been a popular programming language.
       Score: 1 (Correct)
    ---
    Evaluation finished, but no project link was generated.
    ```

## What's Next (Future Work)

* **UI-based LangSmith Evaluation**: Demonstrate how to create and run evaluation experiments directly from the LangSmith UI.
* **More Robust Error Handling**: Enhance error handling within the agent's tools and nodes.
* **Expanded Test Dataset**: Grow the evaluation dataset with more diverse and challenging queries.
* **Agent Improvements**: Explore more advanced LangGraph patterns, such as self-correction loops or integrating additional tools.

## Known Limitations / Friction Log

* **Hardcoded Wikipedia Scope**: The `wikipedia_url_searcher` is strictly limited to Wikipedia. While intentional for this demo, a real-world agent would likely need broader search capabilities.
* **Critic Node Simplicity**: The current CriticNode is quite basic. It simply accepts or rejects based on a simple prompt. More sophisticated critique mechanisms could be explored.
* **Debugging Verbosity**: While `--debug` mode provides useful insights into the agent's flow, the sheer volume of output can sometimes be overwhelming for complex traces. Better structured logging or visualization tools would be beneficial.
* **API Key Management**: Requiring manual editing of `.env` is standard but less convenient for quick starts. Future iterations might consider safer, more automated credential management for demos.
