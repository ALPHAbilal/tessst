"""
Enhanced Intent Determination Module

This module provides sophisticated intent determination capabilities
by leveraging LLM-powered agents rather than regex patterns or hardcoded fallbacks.
"""

import json
import logging
import asyncio
from typing import Dict, List, Tuple, Any, Optional

# Setup logging
logger = logging.getLogger(__name__)

async def get_semantic_intent_scores(user_query: str, client, model: str) -> Dict[str, float]:
    """
    Calculate semantic similarity between query and different intents
    using the LLM's understanding rather than keyword matching.
    """
    prompt = f"""
    Analyze the following user query and rate how strongly it matches each of these intents on a scale of 0.0 to 1.0:
    
    User Query: "{user_query}"
    
    Intents to rate:
    1. populate_template: User wants to fill, create, or generate a document based on a template
    2. analyze_template: User wants to analyze, compare, or explain aspects of a template
    3. kb_query: User wants information from the knowledge base
    4. temp_context_query: User wants information from uploaded documents
    5. kb_query_with_temp_context: User wants to combine knowledge base information with uploaded documents
    
    For each intent, provide a score between 0.0 (not matching at all) and 1.0 (perfect match).
    Explain your reasoning briefly, then provide the scores in JSON format.
    
    Output JSON format:
    {{"populate_template": score, "analyze_template": score, "kb_query": score, "temp_context_query": score, "kb_query_with_temp_context": score}}
    """
    
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        content = response.choices[0].message.content
        scores = json.loads(content)
        
        # Validate scores
        valid_scores = {}
        for intent, score in scores.items():
            if isinstance(score, (int, float)) and 0 <= score <= 1:
                valid_scores[intent] = float(score)
            else:
                logger.warning(f"Invalid score for {intent}: {score}, defaulting to 0.5")
                valid_scores[intent] = 0.5
                
        return valid_scores
    except Exception as e:
        logger.error(f"Error getting semantic intent scores: {e}")
        # Return default scores if there's an error
        return {
            "populate_template": 0.5,
            "analyze_template": 0.5,
            "kb_query": 0.5,
            "temp_context_query": 0.5,
            "kb_query_with_temp_context": 0.5
        }

async def analyze_template_context(template_name: str, template_content: str, user_query: str, client, model: str) -> Dict[str, Any]:
    """
    Analyze template to determine document type and required fields using LLM
    rather than regex patterns.
    """
    prompt = f"""
    Analyze the following template and user query to determine:
    1. The document type (e.g., employment_contract, invoice, general_document)
    2. Required fields that need to be filled in this template
    3. How likely the user wants to populate vs. analyze this template (score 0-1 for each)
    
    Template Name: {template_name}
    
    Template Content:
    {template_content[:2000]}  # Limit content length
    
    User Query: "{user_query}"
    
    Provide your analysis in JSON format with these keys:
    - document_type: string
    - required_fields: array of strings (field names)
    - populate_score: float (0-1)
    - analyze_score: float (0-1)
    """
    
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        content = response.choices[0].message.content
        analysis = json.loads(content)
        
        # Ensure required fields are normalized
        if "required_fields" in analysis and isinstance(analysis["required_fields"], list):
            analysis["required_fields"] = [
                field.lower().replace(" ", "_") if isinstance(field, str) else str(field)
                for field in analysis["required_fields"]
            ]
        
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing template context: {e}")
        # Return minimal default analysis
        return {
            "document_type": "unknown",
            "required_fields": [],
            "populate_score": 0.5,
            "analyze_score": 0.5
        }

async def analyze_conversation_context(history: List[Dict[str, str]], user_query: str, client, model: str) -> Dict[str, float]:
    """
    Analyze conversation history to determine intent continuity and context
    """
    # Extract the last few messages for context
    recent_messages = history[-5:] if len(history) > 5 else history
    
    # Format messages for the prompt
    formatted_history = "\n".join([
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')[:100]}..."
        for msg in recent_messages
    ])
    
    prompt = f"""
    Analyze this conversation history and the new user query to determine:
    1. If the user is continuing a previous intent
    2. If the user is referring to previous outputs
    3. How the conversation context affects the likely intent
    
    Conversation History:
    {formatted_history}
    
    New User Query: "{user_query}"
    
    Rate how the conversation history suggests each intent (0.0 to 1.0):
    - populate_template: User wants to fill a template
    - analyze_template: User wants to analyze a template
    - kb_query: User wants knowledge base information
    - temp_context_query: User wants information from uploaded documents
    - kb_query_with_temp_context: User wants combined information
    
    Provide your analysis as a JSON object with these intents as keys and scores as values.
    """
    
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        content = response.choices[0].message.content
        bias_scores = json.loads(content)
        
        # Validate scores
        valid_scores = {}
        for intent, score in bias_scores.items():
            if isinstance(score, (int, float)) and 0 <= score <= 1:
                valid_scores[intent] = float(score)
            else:
                valid_scores[intent] = 0.0
                
        return valid_scores
    except Exception as e:
        logger.error(f"Error analyzing conversation context: {e}")
        # Return neutral bias if there's an error
        return {
            "populate_template": 0.0,
            "analyze_template": 0.0,
            "kb_query": 0.0,
            "temp_context_query": 0.0,
            "kb_query_with_temp_context": 0.0
        }

async def estimate_analyzer_confidence(analyzer_output: Any) -> float:
    """
    Estimate the confidence of the QueryAnalyzerAgent's intent determination
    """
    # If the output is a string containing "confidence" or "score", try to extract it
    if isinstance(analyzer_output, str) and ("confidence" in analyzer_output.lower() or "score" in analyzer_output.lower()):
        try:
            # Try to find a confidence value in the string
            import re
            confidence_matches = re.findall(r'confidence["\s:]+([0-9.]+)', analyzer_output.lower())
            score_matches = re.findall(r'score["\s:]+([0-9.]+)', analyzer_output.lower())
            
            matches = confidence_matches + score_matches
            if matches:
                # Use the first match
                confidence = float(matches[0])
                # Ensure it's in the range [0, 1]
                return max(0.0, min(1.0, confidence))
        except Exception:
            pass
    
    # Default confidence based on output type
    if isinstance(analyzer_output, dict) and "intent" in analyzer_output:
        # Structured output suggests higher confidence
        return 0.8
    elif hasattr(analyzer_output, "intent") and hasattr(analyzer_output, "details"):
        # Object with proper attributes suggests higher confidence
        return 0.8
    elif isinstance(analyzer_output, str) and len(analyzer_output) > 0:
        # String output suggests medium confidence
        return 0.6
    else:
        # Unknown output suggests low confidence
        return 0.4

async def determine_final_intent(
    analyzer_intent: str,
    analyzer_details: Dict[str, Any],
    user_query: str,
    template_name: Optional[str] = None,
    template_content: Optional[str] = None,
    temp_files_info: Optional[List[Dict]] = None,
    history: Optional[List[Dict[str, str]]] = None,
    client = None,
    model: str = "gpt-4o-mini"
) -> Tuple[str, Dict[str, Any]]:
    """
    Determine the final intent using a sophisticated approach that leverages LLM capabilities
    rather than regex patterns or hardcoded fallbacks.
    """
    logger.info("Starting enhanced intent determination")
    
    # Initialize scores with the analyzer's intent having a base value
    intent_scores = {
        "populate_template": 0.0,
        "analyze_template": 0.0,
        "kb_query": 0.0,
        "temp_context_query": 0.0,
        "kb_query_with_temp_context": 0.0
    }
    
    # 1. Get semantic intent scores directly from LLM
    semantic_scores = await get_semantic_intent_scores(user_query, client, model)
    logger.info(f"Semantic intent scores: {semantic_scores}")
    
    # Add semantic scores to our total scores
    for intent, score in semantic_scores.items():
        intent_scores[intent] += score
    
    # 2. Consider template context if available
    template_analysis = None
    if template_name and template_content:
        template_analysis = await analyze_template_context(
            template_name, template_content, user_query, client, model
        )
        logger.info(f"Template analysis: {template_analysis}")
        
        # Adjust scores based on template analysis
        if "populate_score" in template_analysis:
            intent_scores["populate_template"] += template_analysis["populate_score"]
        
        if "analyze_score" in template_analysis:
            intent_scores["analyze_template"] += template_analysis["analyze_score"]
    
    # 3. Consider conversation history if available
    if history:
        conversation_bias = await analyze_conversation_context(
            history, user_query, client, model
        )
        logger.info(f"Conversation context bias: {conversation_bias}")
        
        # Add conversation bias to our scores
        for intent, bias in conversation_bias.items():
            intent_scores[intent] += bias * 0.5  # Weight conversation context less
    
    # 4. Consider the analyzer's intent with high weight
    analyzer_confidence = await estimate_analyzer_confidence(analyzer_intent)
    logger.info(f"Analyzer confidence for {analyzer_intent}: {analyzer_confidence:.4f}")
    
    # Give the analyzer's intent a significant boost
    if analyzer_intent in intent_scores:
        intent_scores[analyzer_intent] += analyzer_confidence * 1.5
        logger.info(f"Boosted {analyzer_intent} score based on analyzer confidence")
    
    # 5. Consider UI state (template selection, temp files)
    if template_name:
        # Having a template selected slightly increases template-related intents
        intent_scores["populate_template"] += 0.2
        intent_scores["analyze_template"] += 0.1
    
    if temp_files_info:
        # Having temp files slightly increases temp file-related intents
        intent_scores["temp_context_query"] += 0.2
        intent_scores["kb_query_with_temp_context"] += 0.1
    
    # 6. Make final decision
    final_intent = max(intent_scores.items(), key=lambda x: x[1])[0]
    logger.info(f"Final determined intent: {final_intent} (scores: {intent_scores})")
    
    # 7. Prepare detailed intent information
    final_details = analyzer_details.copy()
    
    if final_intent == "populate_template" and template_analysis:
        # Use template analysis for required fields
        if "required_fields" in template_analysis and template_analysis["required_fields"]:
            final_details["required_fields"] = template_analysis["required_fields"]
            final_details["document_type"] = template_analysis.get("document_type", "unknown")
    
    elif final_intent == "analyze_template" and template_name:
        final_details["template_name"] = template_name
        final_details["needs_kb_lookup"] = True
        final_details["needs_temp_files"] = bool(temp_files_info)
    
    elif final_intent == "kb_query_with_temp_context" and temp_files_info:
        final_details["query_topic"] = user_query
        final_details["temp_file_names"] = [f['filename'] for f in temp_files_info]
    
    elif final_intent == "temp_context_query" and temp_files_info:
        final_details["query"] = user_query
        final_details["temp_filenames"] = [f['filename'] for f in temp_files_info]
    
    # Ensure query is always included in details
    if "query" not in final_details:
        final_details["query"] = user_query
    
    return final_intent, final_details

async def record_intent_determination(
    user_query: str,
    determined_intent: str,
    analyzer_intent: str,
    intent_scores: Dict[str, float],
    final_workflow: Optional[str] = None
) -> None:
    """
    Record intent determination for analysis and improvement
    """
    # This would typically write to a database, but for now we'll just log it
    logger.info(f"Intent Determination Record: Query='{user_query[:50]}...', "
                f"Determined={determined_intent}, Analyzer={analyzer_intent}, "
                f"Scores={intent_scores}, Final Workflow={final_workflow}")
    
    # In a real implementation, we would store this in a database for later analysis
    # intent_db.insert({
    #     "timestamp": time.time(),
    #     "user_query": user_query,
    #     "determined_intent": determined_intent,
    #     "analyzer_intent": analyzer_intent,
    #     "intent_scores": intent_scores,
    #     "final_workflow": final_workflow
    # })
