"""
DocumentAnalyzerAgent Module

This module provides a comprehensive agent for document analysis and data extraction
that replaces regex-based approaches with semantic understanding using LLMs.
"""

import json
import logging
import re
from typing import Dict, List, Optional
from pydantic import BaseModel

from agents import Agent, Runner, function_tool, RunContextWrapper

# Setup logging
logger = logging.getLogger(__name__)

# --- Data Models ---
class DocumentSection(BaseModel):
    """Section of a document"""
    name: str
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    content: str

    model_config = {
        "extra": "forbid"
    }

class DocumentMetadata(BaseModel):
    """Metadata about a document"""
    # Define specific fields that might be in metadata
    # Using Optional fields with default values to handle any case
    document_date: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    language: Optional[str] = None
    # Add a field for any other metadata as a simple string
    additional_info: Optional[str] = None

    model_config = {
        "extra": "forbid"
    }

class DocumentStructure(BaseModel):
    """Document structure analysis result"""
    document_type: str
    sections: List[DocumentSection]
    metadata: DocumentMetadata

    model_config = {
        "extra": "forbid"
    }

class ExtractedField(BaseModel):
    """Single extracted field with metadata"""
    field_name: str
    value: Optional[str]
    confidence: float
    source_location: Optional[str]
    alternatives: Optional[List[str]]

    model_config = {
        "extra": "forbid"
    }

class ExtractedDataResult(BaseModel):
    """Collection of extracted fields with rich metadata"""
    fields: List[ExtractedField]
    document_type: str
    status: str = "success"
    error_message: Optional[str] = None

    model_config = {
        "extra": "forbid"
    }

    def to_simple_dict(self) -> Dict[str, Optional[str]]:
        """Convert to simple dictionary format for compatibility with existing code"""
        return {field.field_name: field.value for field in self.fields}

# --- Function Tools ---
@function_tool
async def analyze_document_structure(ctx: RunContextWrapper, document_content: str, document_name: str) -> DocumentStructure:
    """
    Analyze document structure to identify sections, layout, and document type.

    Args:
        document_content: The text content of the document
        document_name: The name of the document (helpful for type inference)

    Returns:
        DocumentStructure object with document type and section information
    """
    logger.info(f"[Tool Call] analyze_document_structure for: {document_name}")

    # Get OpenAI client from context
    client = ctx.context.get("client")
    if not client:
        logger.error("No OpenAI client found in context")
        empty_metadata = DocumentMetadata(additional_info="Configuration error")
        return DocumentStructure(
            document_type="unknown",
            sections=[],
            metadata=empty_metadata
        )

    # Use model to analyze document structure
    prompt = f"""
    Analyze the structure of this document and identify its type, sections, and organization.

    Document Name: {document_name}

    Document Content:
    {document_content[:4000]}  # Limit content length

    Provide a detailed analysis including:
    1. Document type (e.g., employment_contract, invoice, ID_card, general_document)
    2. Main sections and their boundaries
    3. Key metadata about the document

    Return your analysis as a JSON object with these keys:
    - document_type: string
    - sections: array of objects with {{"name": string, "start_index": number, "end_index": number, "content": string}}
    - metadata: object with any relevant document metadata
    """

    try:
        # Call the model
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Use appropriate model
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        # Parse the response
        content = response.choices[0].message.content
        structure_data = json.loads(content)

        # Process sections
        sections = []
        for section_data in structure_data.get("sections", []):
            try:
                sections.append(DocumentSection(
                    name=section_data.get("name", "Unnamed Section"),
                    start_index=section_data.get("start_index"),
                    end_index=section_data.get("end_index"),
                    content=section_data.get("content", "")
                ))
            except Exception as section_error:
                logger.warning(f"Error processing section: {section_error}")
                # Add a simplified section if there's an error
                sections.append(DocumentSection(
                    name="Error Processing Section",
                    content=str(section_data)[:200]
                ))

        # Process metadata
        metadata_dict = structure_data.get("metadata", {})
        metadata = DocumentMetadata(
            document_date=metadata_dict.get("date"),
            author=metadata_dict.get("author"),
            title=metadata_dict.get("title"),
            language=metadata_dict.get("language"),
            additional_info=json.dumps(metadata_dict) if metadata_dict else None
        )

        # Create and return DocumentStructure
        return DocumentStructure(
            document_type=structure_data.get("document_type", "unknown"),
            sections=sections,
            metadata=metadata
        )
    except Exception as e:
        logger.error(f"[Tool Error] Document structure analysis failed: {e}", exc_info=True)
        error_metadata = DocumentMetadata(additional_info=f"Error: {str(e)}")
        return DocumentStructure(
            document_type="unknown",
            sections=[],
            metadata=error_metadata
        )

@function_tool
async def extract_fields_from_document(
    ctx: RunContextWrapper,
    document_content: str,
    required_fields: List[str],
    document_structure: Optional[DocumentStructure] = None
) -> ExtractedDataResult:
    """
    Extract specific fields from document content using semantic understanding.

    Args:
        document_content: The text content of the document
        required_fields: List of field names to extract
        document_structure: Optional document structure from previous analysis

    Returns:
        ExtractedDataResult object with extracted fields and metadata
    """
    logger.info(f"[Tool Call] extract_fields_from_document. Required fields: {required_fields}")

    # Get OpenAI client from context
    client = ctx.context.get("client")
    if not client:
        logger.error("No OpenAI client found in context")
        return ExtractedDataResult(
            fields=[],
            document_type="unknown",
            status="error",
            error_message="Configuration error"
        )

    # Normalize field names
    normalized_fields = [field.lower().replace(" ", "_") for field in required_fields]

    # Prepare document type and structure information
    doc_type = "unknown"
    structure_info = ""
    if document_structure:
        doc_type = document_structure.document_type
        # Create a simplified representation of sections for the prompt
        sections_info = []
        for section in document_structure.sections:
            section_preview = section.content[:200] + "..." if len(section.content) > 200 else section.content
            sections_info.append({"name": section.name, "content": section_preview})

        # Create a simplified representation of metadata
        metadata_dict = {
            "document_date": document_structure.metadata.document_date,
            "author": document_structure.metadata.author,
            "title": document_structure.metadata.title,
            "language": document_structure.metadata.language,
            "additional_info": document_structure.metadata.additional_info
        }
        # Remove None values
        metadata_dict = {k: v for k, v in metadata_dict.items() if v is not None}

        structure_info = f"""
        Document Type: {document_structure.document_type}

        Document Sections:
        {json.dumps(sections_info, indent=2)}

        Document Metadata:
        {json.dumps(metadata_dict, indent=2)}
        """

    # Create field-specific guidelines based on document type
    field_guidelines = []
    for field in normalized_fields:
        if 'date' in field:
            field_guidelines.append(f"- {field}: Look for dates in various formats (DD/MM/YYYY, Month DD, YYYY)")
        elif 'name' in field or 'nom' in field:
            field_guidelines.append(f"- {field}: Look for person names, typically in formats like 'First Last' or 'Last, First'")
        elif 'address' in field or 'adresse' in field:
            field_guidelines.append(f"- {field}: Look for physical addresses, which may span multiple lines")
        elif 'amount' in field or 'montant' in field or 'salary' in field or 'salaire' in field:
            field_guidelines.append(f"- {field}: Look for monetary amounts, possibly with currency symbols")
        elif 'id' in field or 'number' in field or 'numéro' in field:
            field_guidelines.append(f"- {field}: Look for identification numbers, which may have specific formats")
        elif 'employer' in field or 'employeur' in field or 'company' in field or 'société' in field:
            field_guidelines.append(f"- {field}: Look for company or organization names")
        elif 'job' in field or 'title' in field or 'poste' in field or 'fonction' in field:
            field_guidelines.append(f"- {field}: Look for job titles or positions")
        elif 'duration' in field or 'durée' in field or 'period' in field or 'période' in field:
            field_guidelines.append(f"- {field}: Look for time periods or durations")
        else:
            field_guidelines.append(f"- {field}: Extract this field based on context")

    field_guidelines_text = "\n".join(field_guidelines)

    # Get the user query from the context if available
    user_query = ctx.context.get("current_query", "") if hasattr(ctx, "context") else ""

    # Create the prompt
    prompt = f"""
    Extract the following fields from this document using semantic understanding rather than pattern matching.

    Required Fields:
    {field_guidelines_text}

    {structure_info}

    Document Content:
    {document_content[:6000]}  # Limit content length

    User Query: {user_query}

    For each field:
    1. Extract the most likely value
    2. Provide a confidence score (0.0-1.0)
    3. Note where in the document you found it
    4. List alternative values if applicable

    Return your extraction as a JSON array of objects with these keys:
    - field_name: string (normalized field name)
    - value: string or null
    - confidence: number (0.0-1.0)
    - source_location: string (description of where in the document)
    - alternatives: array of strings (alternative values) or null
    """

    try:
        # Call the model
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Use appropriate model
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        # Parse the response
        content = response.choices[0].message.content
        extraction_data = json.loads(content)

        # Process the extracted fields
        extracted_fields = []
        if isinstance(extraction_data, list):
            for field_data in extraction_data:
                extracted_fields.append(
                    ExtractedField(
                        field_name=field_data.get("field_name", "unknown"),
                        value=field_data.get("value"),
                        confidence=field_data.get("confidence", 0.0),
                        source_location=field_data.get("source_location"),
                        alternatives=field_data.get("alternatives")
                    )
                )
        elif isinstance(extraction_data, dict) and "fields" in extraction_data:
            # Handle case where model returns a wrapper object
            for field_data in extraction_data["fields"]:
                extracted_fields.append(
                    ExtractedField(
                        field_name=field_data.get("field_name", "unknown"),
                        value=field_data.get("value"),
                        confidence=field_data.get("confidence", 0.0),
                        source_location=field_data.get("source_location"),
                        alternatives=field_data.get("alternatives")
                    )
                )

        # Ensure all required fields are included
        existing_fields = {field.field_name for field in extracted_fields}
        for field in normalized_fields:
            if field not in existing_fields:
                extracted_fields.append(
                    ExtractedField(
                        field_name=field,
                        value=None,
                        confidence=0.0,
                        source_location=None,
                        alternatives=None
                    )
                )

        # Create and return ExtractedDataResult
        return ExtractedDataResult(
            fields=extracted_fields,
            document_type=doc_type,
            status="success"
        )
    except Exception as e:
        logger.error(f"[Tool Error] Field extraction failed: {e}", exc_info=True)
        # Create empty fields for all required fields
        empty_fields = [
            ExtractedField(
                field_name=field,
                value=None,
                confidence=0.0,
                source_location=None,
                alternatives=None
            )
            for field in normalized_fields
        ]
        return ExtractedDataResult(
            fields=empty_fields,
            document_type=doc_type if doc_type else "unknown",
            status="error",
            error_message=str(e)
        )

@function_tool
async def detect_fields_from_template(ctx: RunContextWrapper, template_content: str, template_name: str) -> List[str]:
    """
    Detect required fields from a template using semantic understanding.

    Args:
        template_content: The text content of the template
        template_name: The name of the template

    Returns:
        List of detected field names
    """
    logger.info(f"[Tool Call] detect_fields_from_template for: {template_name}")

    # Get OpenAI client from context
    client = ctx.context.get("client")
    if not client:
        logger.error("No OpenAI client found in context")
        return []

    # Create the prompt
    prompt = f"""
    Analyze this document template and identify all fields that need to be filled.

    Template Name: {template_name}

    Template Content:
    {template_content[:4000]}  # Limit content length

    Look for:
    1. Explicit placeholders like [Field Name], {{Field Name}}, <Field Name>, etc.
    2. Implied fields based on context (e.g., "The employee, _____, agrees to...")
    3. Standard fields expected in this type of document

    Return a JSON array of field names, normalized to lowercase with underscores instead of spaces.
    """

    try:
        # Call the model
        response = await client.chat.completions.create(
            model="gpt-4o-mini",  # Use appropriate model
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        # Parse the response
        content = response.choices[0].message.content
        fields_data = json.loads(content)

        # Process the fields
        if isinstance(fields_data, list):
            detected_fields = fields_data
        elif isinstance(fields_data, dict) and "fields" in fields_data:
            detected_fields = fields_data["fields"]
        else:
            detected_fields = []
            logger.warning(f"Unexpected fields data format: {fields_data}")

        # Normalize field names
        normalized_fields = [
            field.lower().replace(" ", "_") if isinstance(field, str) else str(field)
            for field in detected_fields
        ]

        # Remove duplicates while preserving order
        unique_fields = []
        for field in normalized_fields:
            if field not in unique_fields:
                unique_fields.append(field)

        logger.info(f"[Tool Result] Detected fields from template: {unique_fields}")
        return unique_fields
    except Exception as e:
        logger.error(f"[Tool Error] Field detection failed: {e}", exc_info=True)
        return []

# --- Define the DocumentAnalyzerAgent ---
document_analyzer_agent = Agent(
    name="DocumentAnalyzerAgent",
    instructions="""You are a specialized document analyzer agent that understands document structure and extracts information using semantic understanding rather than pattern matching.

You have three main capabilities:
1. Analyzing document structure to identify sections, layout, and document type
2. Extracting specific fields from documents based on semantic understanding
3. Detecting required fields from templates

When analyzing documents:
- Focus on understanding the document's semantic structure, not just looking for patterns
- Consider the document type when extracting information
- Provide confidence scores for extracted values
- Suggest alternative values when appropriate

When detecting fields from templates:
- Look for explicit placeholders in various formats
- Identify implied fields based on context
- Consider standard fields expected in the document type

Use the appropriate tool based on the task:
- analyze_document_structure: To understand document organization
- extract_fields_from_document: To extract specific fields
- detect_fields_from_template: To identify fields in a template

Always return structured data according to the tool's output format.
""",
    tools=[analyze_document_structure, extract_fields_from_document, detect_fields_from_template],
    model="gpt-4o-mini"  # Use appropriate model
)

# --- Compatibility Functions for Existing Code ---

async def extract_data_for_template_agent_based(ctx: RunContextWrapper, context_sources: List[str], required_fields: List[str]) -> Dict[str, Optional[str]]:
    """
    Agent-based replacement for the regex-based extract_data_for_template function.

    Args:
        context_sources: List of text content from various sources
        required_fields: List of field names to extract

    Returns:
        Dictionary mapping field names to extracted values
    """
    logger.info(f"[Agent Call] extract_data_for_template_agent_based. Required: {required_fields}. Sources: {len(context_sources)} provided.")

    # Combine context sources
    combined_context = "\n\n".join(context_sources)
    if not combined_context:
        logger.warning("No context provided for extraction")
        return {field: None for field in required_fields}

    try:
        # Create a simple extraction agent with instructions
        extraction_agent = Agent(
            name="DocumentFieldExtractor",
            instructions=f"""Extract the following fields from the document: {', '.join(required_fields)}.
            For each field, provide the value if found in the document, or indicate if it's not found.
            Format your response as a list of field:value pairs, one per line.
            """,
            model="gpt-4o-mini"
        )

        # Create the prompt with the document content
        prompt = f"""Here is the document content to analyze:

        {combined_context[:4000]}  # Limit content length

        Please extract values for these fields: {', '.join(required_fields)}
        """

        # Run the agent using the Runner class (static method)
        result = await Runner.run(extraction_agent, input=prompt, context=ctx.context)

        # Extract the data from the agent's response
        extracted_data = {field: None for field in required_fields}

        # Get the final output from the result
        # The final_output might be a string, dict, or other type
        final_output = result.final_output

        # Convert the final output to a string for pattern matching
        response_text = ""
        if isinstance(final_output, str):
            response_text = final_output
        elif hasattr(final_output, 'markdown_response'):
            response_text = final_output.markdown_response
        elif isinstance(final_output, dict):
            response_text = str(final_output)
        else:
            # Try to convert to string as a fallback
            response_text = str(final_output)

        # Try to find any extracted fields in the agent's response
        for field in required_fields:
            # Look for patterns like "field: value" in the agent's response
            field_pattern = re.compile(f"{field}:\s*([^\n]+)")
            match = field_pattern.search(response_text)
            if match:
                extracted_data[field] = match.group(1).strip()

        # Log the results
        logger.info(f"[Agent Result] Extracted data: {json.dumps(extracted_data, ensure_ascii=False)}")

        return extracted_data
    except Exception as e:
        logger.error(f"[Agent Error] Document analysis failed: {e}", exc_info=True)
        return {field: None for field in required_fields}

async def detect_required_fields_agent_based(template_content: str, template_name: str) -> List[str]:
    """
    Agent-based replacement for the regex-based detect_required_fields_from_template function.

    Args:
        template_content: The text content of the template
        template_name: The name of the template

    Returns:
        List of detected field names
    """
    from app import get_openai_client

    logger.info(f"[Agent Call] detect_required_fields_agent_based for: {template_name}")

    try:
        # Create a context with the OpenAI client
        context = {"client": get_openai_client()}

        # Create a field detection agent
        field_detection_agent = Agent(
            name="TemplateFieldDetector",
            instructions="""Analyze document templates and identify all fields that need to be filled.
            Return a list of field names, normalized to lowercase with underscores instead of spaces.
            Format your response as a bulleted list, with one field per line.
            """,
            model="gpt-4o-mini"
        )

        # Create a prompt for the agent
        prompt = f"""Analyze this document template and identify all fields that need to be filled.

        Template Name: {template_name}

        Template Content:
        {template_content[:4000]}  # Limit content length

        Return a list of field names, normalized to lowercase with underscores instead of spaces.
        """

        # Run the agent using the Runner class (static method)
        result = await Runner.run(field_detection_agent, input=prompt, context=context)

        # Extract the field names from the agent's response
        # Look for a list or JSON structure in the response
        import json
        detected_fields = []

        # Get the final output from the result
        # The final_output might be a string, dict, or other type
        final_output = result.final_output

        # Convert the final output to a string for pattern matching
        response_text = ""
        if isinstance(final_output, str):
            response_text = final_output
        elif hasattr(final_output, 'markdown_response'):
            response_text = final_output.markdown_response
        elif isinstance(final_output, dict):
            response_text = str(final_output)
        else:
            # Try to convert to string as a fallback
            response_text = str(final_output)

        # Try to find a JSON array in the response
        json_pattern = re.compile(r'\[.*\]')
        json_match = json_pattern.search(response_text)
        if json_match:
            try:
                detected_fields = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # If no JSON found, look for field names in the text
        if not detected_fields:
            # Look for patterns like "- field_name" or "* field_name" in the agent's response
            field_pattern = re.compile(r'[\-\*]\s+([a-z0-9_]+)')
            matches = field_pattern.findall(response_text)
            if matches:
                detected_fields = matches

        # Normalize field names
        normalized_fields = [
            field.lower().replace(" ", "_") if isinstance(field, str) else str(field)
            for field in detected_fields
        ]

        # Remove duplicates while preserving order
        unique_fields = []
        for field in normalized_fields:
            if field not in unique_fields:
                unique_fields.append(field)

        logger.info(f"[Agent Result] Detected fields from template: {unique_fields}")
        return unique_fields
    except Exception as e:
        logger.error(f"[Agent Error] Field detection failed: {e}", exc_info=True)
        # Fall back to regex-based detection if agent-based detection fails
        from app import detect_required_fields_from_template
        logger.info("Falling back to regex-based field detection")
        return detect_required_fields_from_template(template_content, template_name)
