import pandas as pd
from llama_index.core import Document
from typing import List
import logging

logger = logging.getLogger(__name__)

def parse_qa_csv(file_path: str) -> List[Document]:
    """
    Parse a CSV file containing Questions and Answers into LlamaIndex Document format.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        List[Document]: List of LlamaIndex Document objects
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV doesn't have required columns
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Validate columns
        required_columns = ['Questions', 'Answers']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        documents = []
        
        # Convert each Q&A pair into a document
        for idx, row in df.iterrows():
            # Format the text in a structured way
            text = f"""
                    Question: {row['Questions']}
                    Answer: {row['Answers']}
                    """
            # Create Document with metadata
            doc = Document(
                text=text.strip(),
                metadata={
                    'question': row['Questions'],
                    'source_type': 'qa_csv',
                    'doc_id': f'qa_{idx}'
                }
            )
            documents.append(doc)
            
        logger.info(f"Successfully parsed {len(documents)} Q&A pairs from CSV")
        return documents
        
    except FileNotFoundError:
        logger.error(f"CSV file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error parsing CSV: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Test the parser with a sample file
    try:
        docs = parse_qa_csv("VWFAQ.csv")
        print(f"Parsed {len(docs)} documents")
        print("\nSample document:")
        print(docs[0].text if docs else "No documents parsed")
    except Exception as e:
        print(f"Error: {str(e)}")