"""
DocumentAnalyzerAgent Integration Module

This module provides integration functions to connect the DocumentAnalyzerAgent
with the existing app.py workflow.
"""

import logging

# Import the shared data model
from data_models import ExtractedData

# Import the required function from DocumentAnalyzerAgent
from document_analyzer_agent import extract_data_for_template_agent_based

# Setup logging
logger = logging.getLogger(__name__)

async def extract_data_for_template_integrated(ctx, context_sources, required_fields):
    """Fixed integration that properly handles the ExtractedData class"""
    logger.info(f"[Integration] extract_data_for_template_integrated called with {len(required_fields)} fields")

    try:
        # Call the agent-based implementation
        extracted_data = await extract_data_for_template_agent_based(ctx, context_sources, required_fields)

        # Ensure all required fields exist, even if null
        for field in required_fields:
            if field not in extracted_data:
                extracted_data[field] = None

        # Return the proper ExtractedData object
        return ExtractedData(
            data=extracted_data,
            status="success",
            error_message=None
        )
    except Exception as e:
        logger.error(f"[Integration Error] extract_data_for_template_integrated failed: {e}", exc_info=True)
        return ExtractedData(
            data={field: None for field in required_fields},
            status="error",
            error_message=str(e)
        )

# detect_required_fields_from_template_integrated removed - now calling detect_fields_from_template tool directly
