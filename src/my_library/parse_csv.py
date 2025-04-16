import pandas as pd
from llama_index.core import Document
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def parse_qa_csv(file_path: str) -> List[Dict[str, str]]:
    """
    Parse a CSV file containing Questions and Answers into a list of dictionaries.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        List[Dict[str, str]]: List of dictionaries with keys 'Id', 'Questions', 'Answers'
        
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
        
        result = []
        
        # Convert each Q&A pair into a pandas DataFrame row
        result = []
        for idx, row in df.iterrows():
            entry = {
                'Id': str(idx + 1),
                'Questions': str(row['Questions']),
                'Answers': str(row['Answers'])
            }
            result.append(entry)
            
        logger.info(f"Successfully parsed {len(result)} Q&A pairs from CSV")
        return result
        
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
        docs = parse_qa_csv("VWFAQ2.csv")
        print(f"Parsed {len(docs)} documents")
        print("\nSample entry:")
        print(docs[0] if docs else "No entries parsed")
    except Exception as e:
        print(f"Error: {str(e)}")