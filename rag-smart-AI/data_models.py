"""
Shared data models for the RAG application.

This module contains Pydantic models that are shared between different
components of the application to avoid circular imports.
"""

from typing import Dict, Optional
from pydantic import BaseModel

class ExtractedData(BaseModel):
    """Data extracted from documents with status information."""
    data: Dict[str, Optional[str]]  # Dictionary with string keys and optional string values
    status: str = "success"
    error_message: Optional[str] = None
