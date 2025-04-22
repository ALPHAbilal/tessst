"""
PyMuPDF helper functions for PDF handling
"""

import os
import subprocess
import tempfile
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Try to import PyMuPDF
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    logger.warning("PyMuPDF (fitz) not available. Will use fallback methods for PDF handling.")
    PYMUPDF_AVAILABLE = False

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file using multiple methods.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text as a string
    """
    # Try PyMuPDF first if available
    if PYMUPDF_AVAILABLE:
        try:
            doc = open_pdf(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}. Trying fallback method.")

    # Fallback to pdftotext if available
    try:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_path = temp_file.name

        # Try using pdftotext (from poppler-utils)
        try:
            subprocess.run(["pdftotext", file_path, temp_path], check=True, capture_output=True)
            with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            os.unlink(temp_path)  # Clean up temp file
            return text
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("pdftotext not available or failed. Trying another fallback.")
            os.unlink(temp_path)  # Clean up temp file

            # Try using pdf2txt.py from pdfminer
            try:
                result = subprocess.run(["pdf2txt.py", file_path], check=True, capture_output=True, text=True)
                return result.stdout
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.warning("pdf2txt.py not available or failed.")

                # Last resort - return a message
                return "[PDF text extraction failed. Please convert the PDF to text manually.]"
    except Exception as e:
        logger.error(f"All PDF extraction methods failed: {e}")
        return "[PDF text extraction failed due to an error.]"

def open_pdf(file_path):
    """
    Open a PDF file using PyMuPDF (fitz) with proper error handling.

    Args:
        file_path: Path to the PDF file

    Returns:
        A PyMuPDF document object
    """
    if not PYMUPDF_AVAILABLE:
        raise ImportError("PyMuPDF (fitz) is not available")

    try:
        # Different versions of PyMuPDF have different APIs
        # Try different methods to open the PDF
        try:
            # Newer versions use fitz.open()
            return fitz.open(file_path)
        except AttributeError:
            # Older versions might use fitz.Document()
            try:
                return fitz.Document(file_path)
            except AttributeError:
                # Some versions use a different import structure
                import pymupdf
                return pymupdf.open(file_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF file: {e}")
