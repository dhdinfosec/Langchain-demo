# LangGraph "Search-then-Read" RAG Agent (Wikipedia-Only Edition)

This repository contains a simple, yet sophisticated, conversational AI agent built using LangGraph. The agent mimics a human research process to answer questions by exclusively utilizing Wikipedia as its external information source. This project also demonstrates how to set up an evaluation pipeline using a local Python script for basic testing.

**Please Note:** This is a work in progress, and the current focus is on the core agent functionality and a basic local evaluation setup. LangSmith integration for comprehensive evaluation will be added in a future update.

## Features

* **Constrained Two-Step RAG ("Search-then-Read")**: The agent employs a specific two-step process:
    * `wikipedia_url_searcher`: A tool (powered by Tavily) that is hard-coded to search only within `en.wikipedia.org` for relevant URLs.
    * `web_page_reader`: A tool that reads the full content of a URL provided by the searcher.
* **Advanced Agentic Reasoning**: The agent's core prompt guides it through this specific research process, allowing it to answer from its own knowledge for common questions or resort to tools for more complex or current information.
* **CriticNode**: A final quality gate that evaluates the agent's synthesized answer and requests revisions if the response is unsatisfactory.
* **Local Evaluation Script**: A Python script (`evaluate.py`) is provided to run a predefined suite of tests against the agent and report pass/fail results based on expected keywords in the answers.

## Project Structure

* `setup.sh`: A bash script to automate the project environment setup, including virtual environment creation and package installation.
* `agent.py`: Contains the core LangGraph agent implementation, defining its state, nodes (Agent, Tool Executor, Critic), and the logic (edges) that connect them.
* `evaluate.py`: A Python script for running automated local evaluations of the agent using a keyword-based assertion approach.
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
    * **`LANGSMITH_API_KEY`**: Although full LangSmith evaluation isn't implemented yet, it's good practice to set this up.
    * **`GOOGLE_API_KEY`**: Required for the `ChatGoogleGenerativeAI` models used by the agent.
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

    You can also run the agent with a `--debug` parameter to see more verbose output about its internal steps:
    ```bash
    python3 agent.py --debug
    ```
    In debug mode, the output will include details about the agent's reasoning and tool execution:
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

### 3. Run the Automated Evaluation

The `evaluate.py` script allows you to test the agent against a predefined dataset.

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

## What's Next (Future Work)

* **LangSmith Evaluation Integration**: Implement programmatic evaluation using the LangSmith SDK, including dataset creation, custom scoring functions (e.g., LLM-as-a-judge for correctness, helpfulness), and running experiments.
* **UI-based LangSmith Evaluation**: Demonstrate how to create and run evaluation experiments directly from the LangSmith UI.
* **More Robust Error Handling**: Enhance error handling within the agent's tools and nodes.
* **Expanded Test Dataset**: Grow the evaluation dataset with more diverse and challenging queries.
* **Agent Improvements**: Explore more advanced LangGraph patterns, such as self-correction loops or integrating additional tools.

## Known Limitations / Friction Log

* **Reliance on Keyword-Based Evaluation**: The current `evaluate.py` uses simple keyword matching, which can be brittle. A more sophisticated evaluation (like LLM-as-a-judge via LangSmith) is necessary for nuanced assessment of answer quality.
* **Hardcoded Wikipedia Scope**: The `wikipedia_url_searcher` is strictly limited to Wikipedia. While intentional for this demo, a real-world agent would likely need broader search capabilities.
* **Critic Node Simplicity**: The current CriticNode is quite basic. It simply accepts or rejects based on a simple prompt. More sophisticated critique mechanisms could be explored.
* **Debugging Verbosity**: While `--debug` mode provides useful useful insights into the agent's flow, the sheer volume of output can sometimes be overwhelming for complex traces. Better structured logging or visualization tools would be beneficial.
* **API Key Management**: Requiring manual editing of `.env` is standard but less convenient for quick starts. Future iterations might consider safer, more automated credential management for demos.
