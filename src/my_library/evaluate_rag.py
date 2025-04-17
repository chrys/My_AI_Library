import os 
import csv
from typing import List, Dict

from my_library.parse_csv import parse_qa_csv
from my_library.utilities import read_config
from my_library.rag_functions import RAGService
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings

from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.evaluation import evaluate
from datasets import Dataset

import opik 
from opik import Opik, track
from opik.evaluation.metrics import (
    Hallucination,
    AnswerRelevance,
    ContextPrecision,
    ContextRecall
)
from opik.evalation import evaluate as opik_evaluate

# --- LangChain components for RAGAS evaluation ---
# RAGAS uses these under the hood for its LLM calls
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def evaluate_ragas(list_of_questions: List[Dict[str, str]], table_name: str):
    # Set up LLM and embedding model
    GOOGLE_API_KEY = read_config('AI KEYS', 'gemini')
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    
    
    Settings.llm = GoogleGenAI(model="models/gemini-2.0-flash-lite")
    Settings.embed_model = GoogleGenAIEmbedding(model="models/text-embedding-004")

    # Initialize RAGService
    rag_service = RAGService(table_name, "gemini")
    if not rag_service.initialized:
        rag_service.initialize()

    # Prepare evaluation data
    eval_data = []
    results_rows = []

    for q in list_of_questions:
        question = q.get("Questions") or q.get("question")
        expected_answer = q.get("Answers") or q.get("answer")
        qid = q.get("Id") or q.get("id")
        generated_answer = rag_service.ask(question)
        eval_data.append({
            "question": question,
            "ground_truth": expected_answer,
            "answer": generated_answer,
            "contexts": [],  # Optionally, you can add retrieved context if available
        })
        results_rows.append({
            "id": qid,
            "question": question,
            "expected answer": expected_answer,
            "generated answer": generated_answer,
        })

    # Run RAGAS evaluation
    # --- Configure RAGAS with Google Models (via LangChain) ---
    ragas_llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-lite", temperature=0)
    ragas_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # Convert eval_data (list of dicts) to a Ragas Dataset
    # Prepare data for Dataset creation
    dataset_dict = {
        "question": [],
        "ground_truth": [],
        "answer": [],
        "contexts": []
    }

    for item in eval_data:
        dataset_dict["question"].append(item["question"])
        dataset_dict["ground_truth"].append(item["ground_truth"])
        dataset_dict["answer"].append(item["answer"])
        dataset_dict["contexts"].append(item["contexts"])

    # Convert to Dataset
    ragas_dataset = Dataset.from_dict(dataset_dict)       
    
    metrics_to_evaluate = [faithfulness, answer_relevancy, context_precision]
    eval_results = evaluate(
        ragas_dataset, 
        metrics=metrics_to_evaluate,
        llm = ragas_llm,
        embeddings = ragas_embeddings,
        )

    # Add metrics to results_rows
    for i, row in enumerate(results_rows):
        row["faithfulness"] = eval_results["faithfulness"][i]
        row["answer_relevancy"] = eval_results["answer_relevancy"][i]
        row["context_precision"] = eval_results["context_precision"][i]

    # Write results to CSV
    with open("results_ragas.csv", "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = [
            "id", "question", "expected answer", "generated answer",
            "faithfulness", "answer_relevancy", "context_precision"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)


# Example usage:
if __name__ == "__main__":
    qa_list = parse_qa_csv("VWFAQ2.csv")
    #evaluate_ragas(qa_list, "vasilias_weddings2")