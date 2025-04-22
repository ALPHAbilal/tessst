"""
Agent-driven semantic caching for vector store queries.

This module provides a sophisticated caching mechanism that uses LLM-based
semantic understanding to determine when queries are equivalent, rather than
relying on exact string matching or regex patterns.
"""

import json
import time
import hashlib
import logging
from typing import Optional, Dict, Any, Union
import os

from agents import Agent, Runner
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class QuerySimilarityResult(BaseModel):
    """Result from the query similarity agent."""
    are_equivalent: bool
    confidence: float
    reasoning: str

class AgentDrivenSemanticCache:
    """
    A cache that uses an LLM-based agent to determine semantic equivalence between queries.
    
    This cache stores vector store search results and can retrieve them based on
    semantic similarity rather than exact string matching.
    """
    
    def __init__(self, cache_size=1000, ttl=600, confidence_threshold=0.85):
        """
        Initialize the semantic cache.
        
        Args:
            cache_size: Maximum number of entries in the cache
            ttl: Time-to-live for cache entries in seconds
            confidence_threshold: Minimum confidence score for semantic equivalence
        """
        self.cache = {}
        self.timestamps = {}
        self.max_size = cache_size
        self.ttl = ttl
        self.confidence_threshold = confidence_threshold
        
        # Initialize the similarity agent
        self.query_similarity_agent = Agent(
            name="QuerySimilarityAgent",
            instructions="""You are a query similarity analyzer specialized in knowledge base queries.
            
            Your task is to compare two queries and determine if they are semantically equivalent,
            meaning they are asking for the same information even if worded differently.
            
            Consider these factors:
            1. If they're asking for the same information or concept
            2. If they reference the same entities, even if using different terms
            3. If they have the same intent and purpose
            4. If they would likely retrieve the same information from a knowledge base
            
            Examples of equivalent queries:
            - "What does the labor code say about vacation days?" ≈ "How many vacation days according to labor law?"
            - "Information about termination notice periods" ≈ "How much notice is required when terminating employment?"
            - "Contract requirements for consultants" ≈ "What needs to be in a consulting contract?"
            
            Examples of non-equivalent queries:
            - "Maternity leave duration" ≠ "Paternity leave rights"
            - "Minimum wage in 2023" ≠ "Minimum wage in 2022"
            - "Employee rights during probation" ≠ "Length of probation period"
            
            Return a JSON object with:
            - are_equivalent: boolean
            - confidence: float (0-1)
            - reasoning: brief explanation of your decision
            """,
            model="gpt-4o-mini",
            output_type=QuerySimilarityResult
        )
    
    def generate_cache_key(self, vector_store_id: str, query: str, filters: Dict, chat_id: str) -> str:
        """
        Generate a cache key from the query parameters.
        
        Args:
            vector_store_id: ID of the vector store
            query: The search query
            filters: Dictionary of filters applied to the search
            chat_id: ID of the chat session
            
        Returns:
            A hash string to use as the cache key
        """
        key_data = {
            'vs_id': vector_store_id,
            'query': query,
            'filters': json.dumps(filters, sort_keys=True) if filters else None,
            'chat_id': chat_id
        }
        return hashlib.md5(json.dumps(key_data).encode()).hexdigest()
    
    async def find_semantic_match(self, new_query: str, vector_store_id: str, 
                                 filters: Dict, chat_id: str) -> Optional[Any]:
        """
        Find a semantically equivalent query in the cache.
        
        Uses the QuerySimilarityAgent to determine if any cached query is
        semantically equivalent to the new query.
        
        Args:
            new_query: The new search query
            vector_store_id: ID of the vector store
            filters: Dictionary of filters applied to the search
            chat_id: ID of the chat session
            
        Returns:
            The cached result if a semantic match is found, None otherwise
        """
        # First check if we have any entries for this chat and vector store
        relevant_entries = []
        for cache_key, cached_entry in self.cache.items():
            # Check basic conditions first (same chat, vector store)
            if (cached_entry['chat_id'] != chat_id or 
                cached_entry['vector_store_id'] != vector_store_id):
                continue
            
            # Check if cache entry is expired
            if time.time() - self.timestamps[cache_key] > self.ttl:
                logger.debug(f"Removing expired cache entry: {cache_key}")
                del self.cache[cache_key]
                del self.timestamps[cache_key]
                continue
            
            relevant_entries.append((cache_key, cached_entry))
        
        if not relevant_entries:
            logger.debug("No relevant cache entries found")
            return None
        
        logger.info(f"Found {len(relevant_entries)} relevant cache entries to check for semantic similarity")
        
        # Now check for semantic equivalence
        for cache_key, cached_entry in relevant_entries:
            # Use agent to check if queries are semantically equivalent
            comparison_prompt = f"""
            Compare these two queries for semantic equivalence:
            
            Query 1 (Cached): {cached_entry['query']}
            Query 2 (New): {new_query}
            
            Document Type Context: {cached_entry.get('document_type', 'general')}
            
            Determine if these queries would likely retrieve the same information from a knowledge base.
            """
            
            try:
                result = await Runner.run(
                    self.query_similarity_agent,
                    input=comparison_prompt
                )
                
                similarity_result = result.final_output
                
                if (similarity_result.are_equivalent and 
                    similarity_result.confidence > self.confidence_threshold):
                    logger.info(f"[SEMANTIC CACHE HIT] Match found with confidence {similarity_result.confidence}")
                    logger.debug(f"Reasoning: {similarity_result.reasoning}")
                    
                    # Update timestamp to keep this entry fresh
                    self.timestamps[cache_key] = time.time()
                    
                    return cached_entry['result']
                else:
                    logger.debug(f"Not equivalent: {similarity_result.reasoning}")
            except Exception as e:
                logger.error(f"Error in query similarity comparison: {e}")
                continue
        
        logger.info("[SEMANTIC CACHE MISS] No semantic match found")
        return None
    
    async def set(self, vector_store_id: str, query: str, filters: Dict, 
                 chat_id: str, document_type: str, result: Any):
        """
        Store a result in the cache.
        
        Args:
            vector_store_id: ID of the vector store
            query: The search query
            filters: Dictionary of filters applied to the search
            chat_id: ID of the chat session
            document_type: Type of document being queried
            result: The result to cache
        """
        # Generate cache key
        cache_key = self.generate_cache_key(vector_store_id, query, filters, chat_id)
        
        # Implement LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            logger.debug(f"Cache full, evicting oldest entry: {oldest_key}")
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        # Store the new entry
        self.cache[cache_key] = {
            'query': query,
            'filters': filters,
            'chat_id': chat_id,
            'vector_store_id': vector_store_id,
            'document_type': document_type,
            'result': result
        }
        self.timestamps[cache_key] = time.time()
        logger.info(f"Added new entry to semantic cache: {query[:50]}...")

# Create global cache instance with configurable settings
semantic_search_cache = AgentDrivenSemanticCache(
    cache_size=int(os.getenv('SEMANTIC_CACHE_SIZE', 1000)),
    ttl=int(os.getenv('SEMANTIC_CACHE_TTL', 600)),
    confidence_threshold=float(os.getenv('SEMANTIC_CACHE_CONFIDENCE_THRESHOLD', 0.85))
)
