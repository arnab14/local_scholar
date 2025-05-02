# data_loader.py
# Handles fetching web content and loading local documents.

import requests
from bs4 import BeautifulSoup
import PyPDF2 # Keep PyPDF2 as a potential fallback
import os
import logging
import fitz # Import fitz (PyMuPDF)

# Use unstructured for potentially better parsing of non-PDF/TXT files
# from unstructured.partition.auto import partition

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_article_text(url: str) -> str | None:
    """Fetches and extracts text content from a given URL."""
    logging.info(f"Attempting to fetch content from URL: {url}")
    try:
        headers = { # Add headers to mimic a browser visit
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        soup = BeautifulSoup(response.content, 'html.parser')

        # Attempt to find main content areas (common tags/attributes)
        main_content = soup.find('article') or soup.find('main') or soup.find(id='content') or soup.find(class_='content') or soup.find(role='main')

        if main_content:
            paragraphs = main_content.find_all('p')
            text = '\n'.join([p.get_text(strip=True) for p in paragraphs])
        else:
             paragraphs = soup.find_all('p')
             text = '\n'.join([p.get_text(strip=True) for p in paragraphs])

        if not text or len(text.split()) < 50:
            logging.warning(f"Paragraph extraction yielded little text for {url}. Falling back to full body text.")
            text = soup.get_text(separator='\n', strip=True)

        text = '\n'.join([line for line in text.splitlines() if line.strip()])
        logging.info(f"Successfully fetched and parsed content from {url}")
        return text

    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error parsing HTML content from {url}: {e}")
        return None

def load_pdf_text_with_pymupdf(file_path: str) -> str | None:
    """Extracts text from a PDF file using PyMuPDF (fitz)."""
    logging.info(f"Attempting PDF parsing with PyMuPDF (fitz): {os.path.basename(file_path)}")
    try:
        # Open the PDF file using fitz
        doc = fitz.open(file_path)
        text = ""
        # Iterate through each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num) # Load the page
            text += page.get_text() + "\n" # Extract text and add a newline separator
        doc.close() # Close the document
        if text:
            logging.info(f"Successfully extracted text from PDF using PyMuPDF: {os.path.basename(file_path)}")
            return text.strip() # Return stripped text if extraction was successful
        else:
            logging.warning(f"PyMuPDF could not extract significant text from PDF: {os.path.basename(file_path)}")
            return None
    except Exception as e:
        logging.error(f"Error during PyMuPDF PDF parsing for {file_path}: {e}")
        return None # Return None if an error occurred


def load_pdf_text_with_pypdf2(file_path: str) -> str | None:
    """Extracts text from a PDF file using PyPDF2 (as fallback)."""
    logging.info(f"Attempting PDF parsing with PyPDF2 (fallback): {os.path.basename(file_path)}")
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            num_warnings = 0
            # Iterate through pages
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text() # Extract text from page
                if page_text:
                    text += page_text + "\n" # Append text and newline
                else:
                    # Limit warnings to avoid flooding logs
                    if num_warnings < 10:
                         logging.warning(f"PyPDF2 could not extract text from page {i+1} of {os.path.basename(file_path)}")
                    elif num_warnings == 10:
                         logging.warning(f"PyPDF2 could not extract text from page {i+1} (further warnings suppressed)...")
                    num_warnings += 1
            if text:
                logging.info(f"Successfully extracted text from PDF using PyPDF2: {os.path.basename(file_path)}")
                return text.strip() # Return stripped text if successful
            else:
                 logging.warning(f"PyPDF2 failed to extract any text from: {os.path.basename(file_path)}")
                 return None
    except Exception as e:
        logging.error(f"Error reading PDF file {file_path} with PyPDF2: {e}")
        return None # Return None on error


def load_txt_text(file_path: str) -> str | None:
    """Reads text from a TXT file."""
    logging.info(f"Reading text from TXT: {os.path.basename(file_path)}")
    try:
        # Open and read the text file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    except Exception as e:
        logging.error(f"Error reading TXT file {file_path}: {e}")
        return None # Return None on error

def load_text_from_file(file_path: str) -> str | None:
    """Loads text from supported local file types, prioritizing PyMuPDF for PDF."""
    logging.info(f"Attempting to load file: {file_path}")
    _, extension = os.path.splitext(file_path) # Get file extension
    extension = extension.lower() # Convert to lowercase

    try:
        if extension == ".pdf":
            # Prioritize PyMuPDF for PDF files
            text = load_pdf_text_with_pymupdf(file_path)
            # Optional: Fallback to PyPDF2 if PyMuPDF fails or returns nothing
            if not text:
                 logging.warning(f"PyMuPDF PDF parsing failed or yielded no text for {os.path.basename(file_path)}. Falling back to PyPDF2.")
                 text = load_pdf_text_with_pypdf2(file_path)
            return text
        elif extension == ".txt":
            # Use the dedicated function for TXT files
            return load_txt_text(file_path)
        else:
            return None
            # Use unstructured for other potential types (DOCX, etc.)
            # Note: May require pip install "unstructured[docx]" etc. later
            # logging.info(f"Attempting to parse {extension} file using 'unstructured'...")
            # elements = partition(filename=file_path) # Use unstructured's auto partition
            # text = "\n".join([el.text for el in elements]) # Combine text from elements
            # if text:
            #     logging.info(f"Successfully parsed {os.path.basename(file_path)} using unstructured.")
            #     return text.strip() # Return stripped text if successful
            # else:
            #     logging.warning(f"'unstructured' could not extract significant text from {os.path.basename(file_path)}")
            #     return None

    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
        return None # Return None if file not found
    except Exception as e:
        # Catch any other exceptions during file loading
        logging.error(f"Error loading file {file_path}: {e}")
        return None # Return None on general error
