"""
DocumentAnalyzerAgent Integration Module

This module provides integration functions to connect the DocumentAnalyzerAgent
with the existing app.py workflow.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union

# Import the DocumentAnalyzerAgent
from document_analyzer_agent import (
    document_analyzer_agent,
    extract_data_for_template_agent_based,
    detect_required_fields_agent_based
)

# Setup logging
logger = logging.getLogger(__name__)

# --- Integration Functions ---

async def extract_data_for_template_integrated(ctx, context_sources, required_fields):
    """
    Integrated version of extract_data_for_template that uses the DocumentAnalyzerAgent
    but maintains compatibility with the existing app.py workflow.

    This function serves as a drop-in replacement for the existing extract_data_for_template function.
    """
    # Import ExtractedData class from app.py
    # We need to import it here to avoid circular imports
    from app import ExtractedData

    logger.info(f"[Integration] extract_data_for_template_integrated called with {len(required_fields)} fields")

    try:
        # Call the agent-based implementation
        extracted_data = await extract_data_for_template_agent_based(ctx, context_sources, required_fields)

        # Ensure we have a dictionary with the required fields
        # If any field is missing, add it with a None value
        for field in required_fields:
            if field not in extracted_data:
                extracted_data[field] = None

        # Create a new ExtractedData instance
        # This is important to ensure we're returning the correct type
        result = ExtractedData(
            data=extracted_data,
            status="success",
            error_message=None
        )

        # Return the ExtractedData object
        return result
    except Exception as e:
        logger.error(f"[Integration Error] extract_data_for_template_integrated failed: {e}", exc_info=True)
        # Return an error ExtractedData object
        empty_data = {field: None for field in required_fields}
        return ExtractedData(data=empty_data, status="error", error_message=f"Data extraction failed: {str(e)}")

async def detect_required_fields_from_template_integrated(template_content, template_name):
    """
    Integrated version of detect_required_fields_from_template that uses the DocumentAnalyzerAgent
    but maintains compatibility with the existing app.py workflow.

    This function serves as a drop-in replacement for the existing detect_required_fields_from_template function.
    """
    logger.info(f"[Integration] detect_required_fields_from_template_integrated called for {template_name}")

    try:
        # Call the agent-based implementation
        detected_fields = await detect_required_fields_agent_based(template_content, template_name)

        # Return the detected fields
        return detected_fields
    except Exception as e:
        logger.error(f"[Integration Error] detect_required_fields_from_template_integrated failed: {e}", exc_info=True)
        # Fall back to the original implementation
        from app import detect_required_fields_from_template
        logger.warning(f"Falling back to original detect_required_fields_from_template for {template_name}")
        return detect_required_fields_from_template(template_content, template_name)
