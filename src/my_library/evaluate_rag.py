import os 
import csv
import datetime

from typing import List, Dict

from my_library.parse_csv import parse_qa_csv
from my_library.utilities import read_config
from my_library.logger import setup_logger
logger = setup_logger()
from my_library.rag_functions import RAGService
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.evaluation import evaluate
from datasets import Dataset

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
  
    
# --- LangChain components for RAGAS evaluation ---
# RAGAS uses these under the hood for its LLM calls
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings 
MY_EMBED_DIMENSION = 1024

def evaluate_ragas(list_of_questions: List[Dict[str, str]], table_name: str, model_type: str):
    # Set up LLM and embedding model
    # if model_type == "gemini":
    #     GOOGLE_API_KEY = read_config('AI KEYS', 'gemini')
    #     os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    #     Settings.llm = GoogleGenAI(model="models/gemini-2.0-flash-lite")
    #     Settings.embed_model = GoogleGenAIEmbedding(model="models/text-embedding-004")
    #     # Initialize RAGService
    #     rag_service = RAGService(table_name, "gemini")
    #     if not rag_service.initialized:
    #         rag_service.initialize()
    # elif model_type == "local":
    #     Settings.llm = Ollama(model="phi3:latest")
    #     Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large:latest")
    #     rag_service = RAGService(table_name, "local")
    #     if not rag_service.initialized:
    #         rag_service.initialize()
    
    rag_service = RAGService(table_name, model_type)
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
    if model_type == "gemini":
        GOOGLE_API_KEY = read_config('AI KEYS', 'gemini')
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        ragas_llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash-lite", temperature=0)
        ragas_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    elif model_type == "local":
        ragas_llm = ChatOllama(model="mistral:latest", temperature=0)
        ragas_embeddings = OllamaEmbeddings(
            model="mistral:latest",)
    else:
        raise ValueError("Unsupported model_type")
    
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
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    results_filename = f".data/results_ragas_{today_str}.csv"
    with open(results_filename, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = [
            "id", "question", "expected answer", "generated answer",
            "faithfulness", "answer_relevancy", "context_precision"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)

import mlflow

def evaluate_mlflow(list_of_questions: List[Dict[str, str]], table_name: str, model_type: str):
    """
    Evaluate a RAG pipeline and log results to MLflow.
    """
    # Set up LLM and embedding model
    if model_type == "gemini":
        GOOGLE_API_KEY = read_config('AI KEYS', 'gemini')
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        Settings.llm = GoogleGenAI(model="models/gemini-2.0-flash-lite")
        Settings.embed_model = GoogleGenAIEmbedding(model="models/text-embedding-004")
        rag_service = RAGService(table_name, "gemini")
        if not rag_service.initialized:
            rag_service.initialize()
    elif model_type == "local":
        # Use Ollama with phi3:latest
        from llama_index.llms.ollama import Ollama
        Settings.llm = Ollama(model="phi3:latest")
        Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large:latest")
        rag_service = RAGService(table_name, "gemini")
        if not rag_service.initialized:
            rag_service.initialize()
    else:
        raise ValueError("Unsupported model_type")

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
            "contexts": [],
        })
        results_rows.append({
            "id": qid,
            "question": question,
            "expected answer": expected_answer,
            "generated answer": generated_answer,
        })

    # Example: Log results to MLflow
    with mlflow.start_run(run_name=f"RAG_Eval_{model_type}_{table_name}"):
        for row in results_rows:
            mlflow.log_metric("faithfulness", 0)  # Placeholder, replace with real metric if available
            mlflow.log_metric("answer_relevancy", 0)
            mlflow.log_metric("context_precision", 0)
            mlflow.log_text(str(row), f"results/{row['id']}.txt")

        # Optionally, log the full results as an artifact
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        results_filename = f"results_mlflow_{today_str}.csv"
        with open(results_filename, "w", newline='', encoding="utf-8") as csvfile:
            fieldnames = [
                "id", "question", "expected answer", "generated answer"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results_rows:
                writer.writerow(row)
        mlflow.log_artifact(results_filename)

    logger.info(f"MLflow run completed. Results saved to {results_filename}")

# Example usage:
if __name__ == "__main__":
    my_table = "vasilias_weddings19Apr25"
    qa_list = parse_qa_csv("./data/VWFAQ4 5 rows.csv")
    evaluate_ragas(qa_list, my_table, "local")