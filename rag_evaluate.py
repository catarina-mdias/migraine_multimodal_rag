import json
import streamlit as st
import pandas as pd
import evaluate as ev
import os

from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, List

from utils.utils import (
    load_from_pickle,
)
from utils.constants import (
    api_key, LLM_MODEL, EMBEDDING_MODEL, BUFFER_DOCS_PATH,
    TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain import hub
from openai import OpenAI

import deepeval
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval import evaluate

# Load metrics
rouge = ev.load("rouge")
meteor = ev.load("meteor")

# Define State TypedDict
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Initialize LLM, Embeddings, Vector Store, Client
llm = ChatOpenAI(model=LLM_MODEL, api_key=api_key)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=api_key)
vector_store = InMemoryVectorStore(embeddings)
client = OpenAI(api_key=api_key)

# Load RAG prompt once
prompt = hub.pull("rlm/rag-prompt")

# Define retrieval and generation functions
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=TOP_K)
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Session state defaults
defaults = {
    "messages": [],  # You can replace with your default_messages
    "input_mode": "text",
    "audio_files": [],
    "audio_mode_locked": False,
    "recording_state": "ready",  # ready, recording, processing
    "current_audio_input": None,
    "last_processed_audio": None,
    "processing_audio": False,
    "show_audio_input": False,
    "indexing": False,
    "graph": None,
    "temp_answer": '',
}

# Initialize session state
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def main_eval():
    st.title("Assistant Evaluator")
    st.markdown("""
    This page allows you to evaluate the assistant.
    Using a set of test questions and answers (in JSON format), evaluate the bot using statistical and LLM evaluation metrics.
    """)

    # Index documents using session state pattern
    if not st.session_state.indexing:
        if not os.path.exists(BUFFER_DOCS_PATH):
            st.warning("⚠️ No documents found. Please go to the **Upload PDFs** page and upload your documents first.")
            st.stop()
        else:
            with st.spinner("Loading..."):
                docs = load_from_pickle(BUFFER_DOCS_PATH)
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    add_start_index=True
                )
                splits = splitter.split_documents(docs)
                vector_store.add_documents(splits)
                graph = StateGraph(State).add_sequence([retrieve, generate])
                graph.add_edge(START, "retrieve")
                st.session_state.graph = graph.compile()
                st.session_state.indexing = True

    # File uploader for Q&A JSON
    uploaded_file = st.file_uploader("Upload a Q&A JSON file", type=["json"])

    if uploaded_file is not None:
        try:
            gt_data = json.load(uploaded_file)
            questions = [item["question"] for item in gt_data]
            references = [item["answer"] for item in gt_data]

            # Generate answers automatically using session state graph
            all_responses = []
            for question in questions:
                response = st.session_state.graph.invoke({"question": question})
                answer = response.get("answer", "") if isinstance(response, dict) else str(response)
                all_responses.append(answer)

            # Display combined results in one table
            df = pd.DataFrame({
                "Question": questions,
                "Ground Truth": references,
                "LLM Answer": all_responses,
            })
            st.subheader("Generated Responses")
            st.dataframe(df, use_container_width=True)

            # Evaluation mode toggle
            eval_mode = st.radio(
                "Choose Evaluation Mode:",
                options=["Statistics-based", "LLM-based"],
                index=0
            )

            if eval_mode == "Statistics-based":
                with st.spinner("Evaluating with Statistics..."):
                    rouge_results = [rouge.compute(predictions=[pred], references=[ref]) for pred, ref in zip(all_responses, references)]
                    rouge_1_scores = [r["rouge1"] for r in rouge_results]
                    rouge_2_scores = [r["rouge2"] for r in rouge_results]
                    meteor_scores = [meteor.compute(predictions=[pred], references=[ref])["meteor"] for pred, ref in zip(all_responses, references)]

                eval_df = pd.DataFrame({
                    "Question": questions,
                    "Ground Truth": references,
                    "LLM Answer": all_responses,
                    "ROUGE-1": rouge_1_scores,
                    "ROUGE-2": rouge_2_scores,
                    "METEOR": meteor_scores,
                })

                st.subheader("Evaluation Results")
                st.dataframe(eval_df, use_container_width=True)

                st.markdown("### Evaluation Summary")
                st.write(f"**Average ROUGE-1:** {sum(rouge_1_scores)/len(rouge_1_scores):.4f}")
                st.write(f"**Average ROUGE-2:** {sum(rouge_2_scores)/len(rouge_2_scores):.4f}")
                st.write(f"**Average METEOR:** {sum(meteor_scores)/len(meteor_scores):.4f}")

            else:  # LLM-based evaluation
                correctness_metric = GEval(
                    name="Correctness",
                    criteria="Determine whether the actual output is factually correct based on the expected output.",
                    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
                )
                with st.spinner("Evaluating with LLM..."):
                    test_cases = [LLMTestCase(input=q, actual_output=pred, expected_output=ref) for q, pred, ref in zip(questions, all_responses, references)]
                    results = evaluate(test_cases=test_cases, metrics=[correctness_metric])
                    scores = [res.metrics_data[0].score for res in results.test_results]

                    eval_df = pd.DataFrame({
                        "Question": [res.input for res in results.test_results],
                        "Ground Truth": [res.expected_output for res in results.test_results],
                        "LLM Answer": [res.actual_output for res in results.test_results],
                        "Success": [res.metrics_data[0].success for res in results.test_results],
                        "Score": scores,
                        "Reason": [res.metrics_data[0].reason for res in results.test_results],
                    })

                    st.subheader("LLM Evaluation Results")
                    st.dataframe(eval_df, use_container_width=True)

                    st.markdown("### Evaluation Summary")
                    st.write(f"**Average Score:** {sum(scores)/len(scores):.4f}")

        except Exception as e:
            st.error(f"Failed to load or process the uploaded JSON file: {e}")
    else:
        st.info("Please upload a JSON file containing questions and answers to begin evaluation.")

if __name__ == '__main__':
    main_eval()