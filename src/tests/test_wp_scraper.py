import unittest
from unittest.mock import patch, MagicMock
import requests
from typing import List, Dict, Any, Optional

# Important: Assuming the script above is saved as 'wp_scraper.py'
from my_library.wp_scraper import fetch_wp_content, clean_html, create_llama_document, scrape_website
from llama_index.core.schema import Document # For type checking

# Sample data for mocking API responses
SAMPLE_POST_1 = {
    "id": 101, "type": "post", "link": "https://fake.com/post1", "title": {"rendered": "Post Title 1 &amp; Stuff"},
    "content": {"rendered": "<p>This is the <strong>first</strong> post content.</p>"}, "date": "2023-01-01T10:00:00"
}
SAMPLE_POST_2 = {
    "id": 102, "type": "post", "link": "https://fake.com/post2", "title": {"rendered": "Post Title 2"},
    "content": {"rendered": "<div>Second post content here.</div>"}, "date": "2023-01-02T11:00:00"
}
SAMPLE_PAGE_1 = {
    "id": 201, "type": "page", "link": "https://fake.com/page1", "title": {"rendered": "Page Title 1"},
    "content": {"rendered": "<h1>Page Content</h1><p>Some text.</p>"}, "date": "2023-01-03T12:00:00"
}
SAMPLE_EMPTY_CONTENT = {
    "id": 301, "type": "post", "link": "https://fake.com/empty", "title": {"rendered": "Empty"},
    "content": {"rendered": ""}, "date": "2023-01-04T13:00:00"
}
SAMPLE_MISSING_CONTENT_KEY = {
    "id": 302, "type": "post", "link": "https://fake.com/no-content-key", "title": {"rendered": "No Content Key"},
    "content": {}, "date": "2023-01-05T14:00:00"
}

class TestFetchWPContent(unittest.TestCase):

    @patch('my_library.wp_scraper.requests.get')
    def test_fetch_success_single_page(self, mock_get):
        # Mock response for a single page of results
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [SAMPLE_POST_1, SAMPLE_POST_2]
        mock_get.return_value = mock_response

        base_url = "https://fake.com"
        content_type = "posts"
        results = fetch_wp_content(base_url, content_type, per_page=10) # per_page > results

        mock_get.assert_called_once_with(
            f"{base_url}/wp-json/wp/v2/{content_type}",
            params={'per_page': 10, 'page': 1, '_embed': ''},
            timeout=30
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['id'], SAMPLE_POST_1['id'])
        self.assertEqual(results[1]['id'], SAMPLE_POST_2['id'])

    @patch('my_library.wp_scraper.requests.get')
    def test_fetch_success_multiple_pages(self, mock_get):
        # Mock responses for two pages of results
        mock_response_page1 = MagicMock()
        mock_response_page1.status_code = 200
        mock_response_page1.json.return_value = [SAMPLE_POST_1] # Page 1 has 1 item

        mock_response_page2 = MagicMock()
        mock_response_page2.status_code = 200
        mock_response_page2.json.return_value = [SAMPLE_POST_2] # Page 2 has 1 item

        mock_response_page3 = MagicMock()
        mock_response_page3.status_code = 200
        mock_response_page3.json.return_value = [] # Page 3 is empty

        # side_effect allows specifying return values for consecutive calls
        mock_get.side_effect = [mock_response_page1, mock_response_page2, mock_response_page3]

        base_url = "https://fake.com"
        content_type = "posts"
        results = fetch_wp_content(base_url, content_type, per_page=1) # Fetch 1 per page

        self.assertEqual(mock_get.call_count, 3) # Called for page 1, 2, and 3 (empty)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['id'], SAMPLE_POST_1['id'])
        self.assertEqual(results[1]['id'], SAMPLE_POST_2['id'])
        # Check params of the last call (page 3)
        mock_get.assert_called_with(
             f"{base_url}/wp-json/wp/v2/{content_type}",
             params={'per_page': 1, 'page': 3, '_embed': ''},
             timeout=30
        )

    @patch('my_library.wp_scraper.requests.get')
    def test_fetch_http_error(self, mock_get):
        # Mock an HTTP error (e.g., 404 Not Found)
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error")
        mock_get.return_value = mock_response

        results = fetch_wp_content("https://fake.com", "posts")
        self.assertEqual(results, []) # Should return empty list on error

    @patch('my_library.wp_scraper.requests.get')
    def test_fetch_request_exception(self, mock_get):
        # Mock a connection error
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        results = fetch_wp_content("https://fake.com", "posts")
        self.assertEqual(results, [])

    @patch('my_library.wp_scraper.requests.get')
    def test_fetch_invalid_json(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Decoding JSON has failed") # Simulate json error
        mock_get.return_value = mock_response

        results = fetch_wp_content("https://fake.com", "posts")
        self.assertEqual(results, [])

    def test_fetch_invalid_content_type(self):
         results = fetch_wp_content("https://fake.com", "comments") # Invalid type
         self.assertEqual(results, [])


class TestCleanHtml(unittest.TestCase):

    def test_clean_basic_html(self):
        html = "<p>Hello <b>World</b>!</p>"
        expected = "Hello World !"
        self.assertEqual(clean_html(html), expected)

    def test_clean_with_links_and_attributes(self):
        html = 'Check <a href="#">this link</a> out.'
        expected = "Check this link out."
        self.assertEqual(clean_html(html), expected)

    def test_clean_multiple_spaces_and_newlines(self):
        html = " Text with \n multiple   spaces \t and tabs. "
        expected = "Text with multiple spaces and tabs." # strip=True removes leading/trailing
        self.assertEqual(clean_html(html), expected)

    def test_clean_html_entities(self):
        html = "Title &amp; stuff"
        expected = "Title & stuff" # BeautifulSoup handles basic entities
        self.assertEqual(clean_html(html), expected)

    def test_clean_empty_string(self):
        self.assertEqual(clean_html(""), "")

    def test_clean_none_input(self):
        self.assertEqual(clean_html(None), "")


class TestCreateLlamaDocument(unittest.TestCase):

    def test_create_document_success(self):
        doc = create_llama_document(SAMPLE_POST_1)
        self.assertIsNotNone(doc)
        self.assertIsInstance(doc, Document)
        # Check text cleaning
        self.assertEqual(doc.text, "This is the first post content.")
        # Check metadata
        self.assertEqual(doc.metadata.get("source_url"), SAMPLE_POST_1["link"])
        self.assertEqual(doc.metadata.get("title"), "Post Title 1 & Stuff") # Note: clean_html used on title
        self.assertEqual(doc.metadata.get("publish_date"), SAMPLE_POST_1["date"])
        self.assertEqual(doc.metadata.get("wp_id"), str(SAMPLE_POST_1["id"]))
        self.assertEqual(doc.metadata.get("type"), SAMPLE_POST_1["type"])
        self.assertEqual(doc.doc_id, f"wp_post_{SAMPLE_POST_1['id']}")


    def test_create_document_page_success(self):
         doc = create_llama_document(SAMPLE_PAGE_1)
         self.assertIsNotNone(doc)
         self.assertIsInstance(doc, Document)
         self.assertEqual(doc.text, "Page Content Some text.")
         self.assertEqual(doc.metadata.get("type"), "page")
         self.assertEqual(doc.doc_id, f"wp_page_{SAMPLE_PAGE_1['id']}")

    def test_create_document_empty_content(self):
        # content['rendered'] is ""
        doc = create_llama_document(SAMPLE_EMPTY_CONTENT)
        self.assertIsNone(doc)

    def test_create_document_missing_content_key(self):
        # content['rendered'] is missing entirely
        doc = create_llama_document(SAMPLE_MISSING_CONTENT_KEY)
        self.assertIsNone(doc)

    def test_create_document_missing_other_keys(self):
        # Test robustness if other keys like title or date are missing
        data = {"id": 500, "content": {"rendered": "<p>Text</p>"}} # Missing title, link, date
        doc = create_llama_document(data)
        self.assertIsNotNone(doc)
        self.assertEqual(doc.text, "Text")
        self.assertEqual(doc.metadata.get("title"), "No Title") # Default value
        self.assertEqual(doc.metadata.get("source_url"), "")
        self.assertEqual(doc.metadata.get("publish_date"), "")
        self.assertEqual(doc.metadata.get("wp_id"), "500")


class TestScrapeWebsite(unittest.TestCase):

    # Patch the functions that are called *by* scrape_website
    @patch('my_library.wp_scraper.fetch_wp_content')
    @patch('my_library.wp_scraper.create_llama_document')
    def test_scrape_success(self, mock_create_doc, mock_fetch):
        # Configure mock_fetch to return different data for posts and pages
        mock_fetch.side_effect = [
            [SAMPLE_POST_1, SAMPLE_POST_2], # Return value for fetch_wp_content(..., 'posts')
            [SAMPLE_PAGE_1]                # Return value for fetch_wp_content(..., 'pages')
        ]

        # Configure mock_create_doc to return mock Documents
        mock_doc1 = MagicMock(spec=Document)
        mock_doc2 = MagicMock(spec=Document)
        mock_doc3 = MagicMock(spec=Document)
        mock_create_doc.side_effect = [mock_doc1, mock_doc2, mock_doc3]

        base_url = "https://fake.com"
        documents = scrape_website(base_url)

        # Assertions
        self.assertEqual(mock_fetch.call_count, 2)
        mock_fetch.assert_any_call(base_url, 'posts')
        mock_fetch.assert_any_call(base_url, 'pages')

        self.assertEqual(mock_create_doc.call_count, 3) # Called for 2 posts, 1 page
        mock_create_doc.assert_any_call(SAMPLE_POST_1)
        mock_create_doc.assert_any_call(SAMPLE_POST_2)
        mock_create_doc.assert_any_call(SAMPLE_PAGE_1)

        self.assertEqual(len(documents), 3)
        self.assertIn(mock_doc1, documents)
        self.assertIn(mock_doc2, documents)
        self.assertIn(mock_doc3, documents)

    @patch('my_library.wp_scraper.fetch_wp_content')
    @patch('my_library.wp_scraper.create_llama_document')
    def test_scrape_fetch_fails(self, mock_create_doc, mock_fetch):
        # Simulate fetch failing entirely
        mock_fetch.return_value = [] # Both calls return empty list

        documents = scrape_website("https://fake.com")

        self.assertEqual(mock_fetch.call_count, 2)
        self.assertEqual(mock_create_doc.call_count, 0) # create_llama_document should not be called
        self.assertEqual(len(documents), 0)

    @patch('my_library.wp_scraper.fetch_wp_content')
    @patch('my_library.wp_scraper.create_llama_document')
    def test_scrape_some_docs_fail_creation(self, mock_create_doc, mock_fetch):
        mock_fetch.side_effect = [
            [SAMPLE_POST_1, SAMPLE_EMPTY_CONTENT], # Posts: one valid, one empty
            [SAMPLE_PAGE_1]                       # Pages: one valid
        ]

        # Simulate create_llama_document returning None for the empty content
        mock_doc1 = MagicMock(spec=Document)
        mock_doc3 = MagicMock(spec=Document)
        mock_create_doc.side_effect = [mock_doc1, None, mock_doc3]

        documents = scrape_website("https://fake.com")

        self.assertEqual(mock_fetch.call_count, 2)
        self.assertEqual(mock_create_doc.call_count, 3) # Called 3 times
        # Check call args for create_llama_document
        mock_create_doc.assert_any_call(SAMPLE_POST_1)
        mock_create_doc.assert_any_call(SAMPLE_EMPTY_CONTENT)
        mock_create_doc.assert_any_call(SAMPLE_PAGE_1)

        self.assertEqual(len(documents), 2) # Only 2 documents were successfully created
        self.assertIn(mock_doc1, documents)
        self.assertIn(mock_doc3, documents)

# --- Test Execution ---

if __name__ == '__main__':
    # Make sure required libraries are installed for testing: pip install llama-index beautifulsoup4 requests
    # Run tests using: python -m unittest test_wp_scraper.py
    unittest.main()