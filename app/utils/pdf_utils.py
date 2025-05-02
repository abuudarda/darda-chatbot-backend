import re
from typing import List
import PyPDF2

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text
    """
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
    
    return text

def chunk_text(text: str) -> List[str]:
    """
    Split text into meaningful chunks
    
    Args:
        text (str): Text to chunk
        
    Returns:
        List[str]: List of text chunks
    """
    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Simple chunking by paragraphs
    # In a production app, consider more sophisticated chunking strategies
    chunks = []
    paragraphs = re.split(r'\n\n+', text)
    
    for para in paragraphs:
        if len(para.strip()) > 50:  # Only keep substantive paragraphs
            chunks.append(para.strip())
    
    return chunks