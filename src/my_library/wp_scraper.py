#!/usr/bin/env python3
import requests
import argparse
import logging
from typing import List, Dict, Optional, Any
from bs4 import BeautifulSoup
from llama_index.core.schema import Document
import sys # For exiting on critical errors

# Configure logging
from my_library.logger import setup_logger
logging = setup_logger()


def fetch_wp_content(base_url: str, content_type: str, per_page: int = 100) -> List[Dict[str, Any]]:
    """
    Fetches posts or pages from a WordPress REST API endpoint with pagination.

    Args:
        base_url: The base URL of the WordPress site (e.g., "https://example.com").
        content_type: The type of content to fetch ('posts' or 'pages').
        per_page: How many items to fetch per request.

    Returns:
        A list of dictionaries, where each dictionary represents a post or page.
        Returns an empty list if fetching fails.
    """
    if content_type not in ['posts', 'pages']:
        logging.error(f"Invalid content_type: {content_type}. Must be 'posts' or 'pages'.")
        return []

    api_endpoint = f"{base_url.rstrip('/')}/wp-json/wp/v2/{content_type}"
    all_items = []
    page = 1

    logging.info(f"Starting fetch for {content_type} from {base_url}...")

    while True:
        params = {'per_page': per_page, 'page': page, '_embed': ''} # _embed can sometimes get featured media/author info
        try:
            response = requests.get(api_endpoint, params=params, timeout=30) # Increased timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

            items = response.json()

            if not items:
                logging.info(f"No more {content_type} found on page {page}.")
                break  # No more items found

            all_items.extend(items)
            logging.info(f"Fetched {len(items)} {content_type} on page {page}.")

            # Simple pagination check: if fewer items than per_page were returned, it's likely the last page
            if len(items) < per_page:
                logging.info("Assuming last page reached based on item count.")
                break

            page += 1

        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching {content_type} page {page} from {api_endpoint}: {e}")
            # Decide if you want to stop entirely or just skip this page/type
            # For this script, we'll stop fetching this content type on error.
            return [] # Return what we have so far, or empty if first page failed
        except ValueError as e: # Catches JSONDecodeError
             logging.error(f"Error decoding JSON for {content_type} page {page} from {api_endpoint}: {e}")
             return []

    logging.info(f"Finished fetching. Total {len(all_items)} {content_type} found.")
    return all_items


def clean_html(html_content: Optional[str]) -> str:
    """
    Removes HTML tags from a string and normalizes whitespace.

    Args:
        html_content: The HTML string to clean.

    Returns:
        The cleaned text content. Returns an empty string if input is None or empty.
    """
    if not html_content:
        return ""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        logging.warning(f"Error cleaning HTML: {e}. Input: {html_content[:100]}...")
        return "" # Return empty string on parsing error


def create_llama_document(item_data: Dict[str, Any]) -> Optional[Document]:
    """
    Creates a LlamaIndex Document object from a WordPress item dictionary.

    Args:
        item_data: A dictionary representing a single WordPress post or page.

    Returns:
        A LlamaIndex Document object with text and metadata, or None if essential data is missing or content is empty.
    """
    try:
        html_content = item_data.get('content', {}).get('rendered')
        title = item_data.get('title', {}).get('rendered', 'No Title')
        link = item_data.get('link', '')
        publish_date = item_data.get('date', '') # Or 'date_gmt'

        if not html_content:
            logging.warning(f"Skipping item ID {item_data.get('id', 'N/A')} due to missing content['rendered']. Link: {link}")
            return None

        cleaned_text = clean_html(html_content)

        if not cleaned_text:
            logging.warning(f"Skipping item ID {item_data.get('id', 'N/A')} due to empty content after cleaning. Link: {link}")
            return None

        metadata = {
            "source_url": link,
            "title": clean_html(title), # Clean title just in case it has HTML entities
            "publish_date": publish_date,
            "wp_id": str(item_data.get('id', '')), # Add WP ID as string
            "type": item_data.get('type', '') # Add type (post/page)
        }

        # Use WP ID as document ID for potential future reference/updates
        doc_id = f"wp_{item_data.get('type', 'item')}_{item_data.get('id', 'unknown')}"

        return Document(
            text=cleaned_text,
            metadata=metadata,
            doc_id=doc_id # Assign the generated doc_id
            )

    except Exception as e:
        logging.error(f"Error creating Document for item ID {item_data.get('id', 'N/A')}: {e}")
        return None


def scrape_website(base_url: str) -> List[Document]:
    """
    Scrapes posts and pages from a WordPress site and converts them to LlamaIndex Documents.

    Args:
        base_url: The base URL of the WordPress site.

    Returns:
        A list of LlamaIndex Document objects.
    """
    logging.info(f"Starting scraping process for {base_url}")
    all_documents: List[Document] = []

    # Fetch Posts
    posts_data = fetch_wp_content(base_url, 'posts')
    for item in posts_data:
        doc = create_llama_document(item)
        if doc:
            all_documents.append(doc)

    # Fetch Pages
    pages_data = fetch_wp_content(base_url, 'pages')
    for item in pages_data:
        doc = create_llama_document(item)
        if doc:
            all_documents.append(doc)

    logging.info(f"Scraping finished. Total {len(all_documents)} documents created.")
    return all_documents


# --- Main Execution ---

def main():
    """Main function to parse arguments and run the scraper."""
    parser = argparse.ArgumentParser(description="Scrape posts and pages from a WordPress site using its REST API.")
    parser.add_argument("url", help="The base URL of the WordPress website (e.g., https://example.com)")
    args = parser.parse_args()

    # Basic URL validation
    if not args.url.startswith(('http://', 'https://')):
        logging.error("Invalid URL format. Please include http:// or https://")
        sys.exit(1) # Exit with error code

    documents = scrape_website(args.url)

    if documents:
        print(f"\nSuccessfully created {len(documents)} LlamaIndex Documents.")
        print("\nSample Document 1:")
        print(f"  Text: {documents[0].text[:200]}...") # Print beginning of text
        print(f"  Metadata: {documents[0].metadata}")
        if documents[0].doc_id:
             print(f"  Doc ID: {documents[0].doc_id}")
        if len(documents) > 1:
            print("\nSample Document 2 Metadata:")
            print(f"  Metadata: {documents[-1].metadata}") # Print metadata of last document
    else:
        print("\nNo documents were created. Check logs for errors.")
        sys.exit(1) # Exit with error code if no documents created

if __name__ == "__main__":
    # Make sure LlamaIndex is installed: pip install llama-index beautifulsoup4 requests
    try:
        from llama_index.core.schema import Document
    except ImportError:
        print("Error: Required libraries not found.")
        print("Please install them using: pip install llama-index beautifulsoup4 requests")
        sys.exit(1)
    main()