# ==============================================================================
#
# LangGraph RAG Agent with a Critic
#
# This script implements a conversational agent that uses a full
# Retrieval-Augmented Generation (RAG) pipeline to answer questions.
#
# Key Components:
# 1. RAG Pipeline: Instead of a simple tool, the agent now uses a multi-step
#    process:
#    a. Formulate Query: Intelligently decide what to search for on Wikipedia.
#    b. Retrieve: Fetch relevant documents from Wikipedia.
#    c. Chunk & Embed: Split documents and create a semantic vector store.
#    d. Synthesize: Generate a final answer based on the retrieved context AND
#       the previous critique.
# 2. AgentNode: Encapsulates the entire RAG pipeline.
# 3. CriticNode: Evaluates the final, synthesized answer for quality.
#
# ==============================================================================

# --- Core Imports ---
import os
import sys
import logging
from datetime import datetime
from typing import List, TypedDict

# --- Third-party Library Imports ---
from dotenv import load_dotenv
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, END

try:
    from termcolor import colored
except ImportError:
    def colored(text, *args, **kwargs): return text
    logging.warning("'termcolor' not found; install for colored output.")

# ==============================================================================
# SCRIPT SETUP AND CONFIGURATION
# ==============================================================================
debug_mode = '--debug' in sys.argv
logging.basicConfig(filename='debug.log', level=logging.DEBUG if debug_mode else logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "LangGraph RAG Agent")

AGENT_MODEL = "gemini-2.0-flash"
CRITIC_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/embedding-001"

google_key = os.getenv("GOOGLE_API_KEY")
if not google_key:
    raise RuntimeError("GOOGLE_API_KEY not set in environment or .env file.")

# ==============================================================================
# RAG PIPELINE COMPONENTS (Initialized Globally)
# ==============================================================================
retriever = WikipediaRetriever(lang="en", load_max_docs=3, doc_content_chars_max=10000)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=google_key)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ==============================================================================
# GRAPH STATE, NODES, AND LOGIC
# ==============================================================================
class AgentState(TypedDict):
    """Represents the state of the agent workflow."""
    messages: List[BaseMessage]
    critique: str
    iterations: int
    today: str

class AgentNode:
    """The node that encapsulates the RAG pipeline."""
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    def _format_docs(self, docs: List[Document]) -> str:
        """Helper function to format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def run(self, state: AgentState):
        """Executes the RAG pipeline."""
        if debug_mode:
            print(colored(f"\nAGENT NODE ITERATION {state['iterations'] + 1}", 'magenta'))
        
        critique = state.get("critique", "")
        
        # --- 1. QUERY FORMULATION ---
        if state['iterations'] == 0:
            query_to_search = state['messages'][0].content
        else:
            if debug_mode:
                print(colored(f"Critique received: {critique}", 'yellow'))
            # --- FIX: Improved prompt to generate better search terms ---
            reformulate_prompt = ChatPromptTemplate.from_template(
                """You are a Wikipedia search expert. Your goal is to formulate the best possible search query to find an answer to the user's question, especially after a failed attempt.
Based on the original query and the critique from the last attempt, identify the key entities (people, concepts, etc.) and create a concise search query that is likely to be a Wikipedia page title.
Do not use full questions. Do not use quotes.

Original Query: {original_query}
Critique: {critique}

Best Wikipedia Search Term:"""
            )
            query_formatter = reformulate_prompt | self.llm | StrOutputParser()
            query_to_search = query_formatter.invoke({
                "original_query": state['messages'][0].content,
                "critique": critique
            })
        if debug_mode:
            print(colored(f"Formulated Search Query: '{query_to_search}'", 'cyan'))

        # --- 2. RETRIEVAL and 3. CHUNKING/EMBEDDING ---
        if debug_mode:
            print(colored("Retrieving documents from Wikipedia...", "blue"))
        docs = retriever.invoke(query_to_search)
        
        context = ""
        if docs:
            if debug_mode:
                print(colored("Chunking and embedding documents...", "blue"))
            chunks = text_splitter.split_documents(docs)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            similar_chunks = vectorstore.similarity_search(state['messages'][0].content, k=5)
            context = self._format_docs(similar_chunks)
        else:
            if debug_mode:
                print(colored("WikipediaRetriever returned no documents for this query.", "yellow"))

        # --- 4. SYNTHESIS (GENERATION) ---
        if debug_mode:
            print(colored("Synthesizing final answer...", "blue"))
        
        # --- FIX: The synthesis prompt now considers the critique as a source of information ---
        synthesis_prompt = ChatPromptTemplate.from_template(
            """You are an expert synthesizer. Your task is to answer the user's question based on the information available.
You have two sources of information:
1.  Retrieved context from a Wikipedia search.
2.  A critique from a previous failed attempt, which may contain the correct answer.

Instructions:
- Prioritize the information in the 'Critique' if it is available and directly answers the question.
- If the critique is not helpful, answer the question based *only* on the provided 'Retrieved Context'.
- If neither source contains the answer, state that you could not find the information.
- Be concise and clear in your final answer.

Today's Date: {today}

Critique from previous attempt:
{critique}

Retrieved Context:
{context}

Question:
{question}"""
        )
        synthesis_chain = (
            synthesis_prompt
            | self.llm
            | StrOutputParser()
        )
        final_answer = synthesis_chain.invoke({
            "context": context,
            "question": state['messages'][0].content,
            "today": state['today'],
            "critique": critique
        })
        
        return {"messages": state["messages"] + [AIMessage(content=final_answer)]}

class CriticNode:
    """The node responsible for evaluating the agent's synthesized answer."""
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.prompt_template = ChatPromptTemplate.from_template(
"""You are an AI critic. Your job is to evaluate if the agent's last response accurately and completely answers the original user query, given the current date.

Today's Date: {today}

- If the response is good and factually correct for the given date, write `ACCEPT`.
- If the response is bad (e.g., inaccurate for the given date, incomplete, or says it can't answer), write `REVISE` and provide a concise reason why, so the agent can reformulate its search.

Original User Query: {original_query}
Agent's Final Answer: {agent_answer}"""
        )

    def run(self, state: AgentState):
        """Runs the critic logic."""
        chain = self.prompt_template | self.llm | StrOutputParser()
        critique = chain.invoke({
            "original_query": state['messages'][0].content,
            "agent_answer": state['messages'][-1].content,
            "today": state['today']
        })
        if debug_mode:
            print(colored(f"Critic response: {critique}", 'yellow'))
        return {"critique": critique, "iterations": state["iterations"] + 1}

def should_continue(state: AgentState) -> str:
    """Determines the next step in the workflow."""
    if state["iterations"] > 3:
        print(colored("--- Too many iterations, ending workflow ---", "red"))
        return END
    
    if state.get("critique", "").strip().upper().startswith("REVISE"):
        return "agent"
    else:
        return END

# ==============================================================================
# GRAPH COMPILATION (Global Scope)
# ==============================================================================
agent_node = AgentNode(llm=ChatGoogleGenerativeAI(model=AGENT_MODEL, temperature=0, google_api_key=google_key))
critic_node = CriticNode(llm=ChatGoogleGenerativeAI(model=CRITIC_MODEL, temperature=0, google_api_key=google_key))

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node.run)
graph_builder.add_node("critic", critic_node.run)

graph_builder.set_entry_point("agent")
graph_builder.add_edge("agent", "critic")
graph_builder.add_conditional_edges("critic", should_continue, {"agent": "agent", END: END})

app = graph_builder.compile()

# ==============================================================================
# MAIN EXECUTION BLOCK (for interactive chat)
# ==============================================================================
def main():
    """Sets up and runs the interactive RAG agent chat."""
    print(colored("Welcome to the RAG-powered AI Agent!", "cyan"))
    print("Type your question (or 'exit' to quit):")

    while True:
        try:
            q = input(colored("Your question: ", "green")).strip()
            if not q or q.lower() == "exit":
                print(colored("\nGoodbye!", "cyan"))
                sys.exit(0)

            today_str = datetime.now().strftime("%Y-%m-%d")

            initial_state: AgentState = {
                "messages": [HumanMessage(content=q)],
                "critique": "",
                "iterations": 0,
                "today": today_str,
            }
            
            final_state = app.invoke(initial_state)
            final_answer = final_state["messages"][-1].content
            
            print(colored("\n--- Final Answer ---", "cyan"))
            print(colored(final_answer, "white"))
            print()
        
        except KeyboardInterrupt:
            print(colored("\nGoodbye!", "cyan"))
            sys.exit(0)
        except Exception as e:
            print(colored(f"\nAn unexpected error occurred: {e}", "red"))
            break

if __name__ == "__main__":
    main()
