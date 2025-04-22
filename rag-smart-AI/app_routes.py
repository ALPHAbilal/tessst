import logging
import asyncio
import json
from flask import jsonify, request
from typing import Dict, List, Optional, Any, Union

# Setup logging
logger = logging.getLogger(__name__)

# Import agent system
from agent_integration import agent_system

async def api_agent_check_template(app):
    """API endpoint to check if a template exists"""
    @app.route('/api/agent/check_template', methods=['GET'])
    async def agent_check_template():
        template_name = request.args.get('name', '')
        if not template_name:
            return jsonify({"error": "No template name provided"}), 400
            
        if not agent_system:
            return jsonify({"error": "Agent system not available"}), 500
            
        try:
            result = await agent_system.check_template_exists(template_name)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error checking template existence: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

async def api_agent_check_kb(app):
    """API endpoint to check if a knowledge base exists"""
    @app.route('/api/agent/check_kb', methods=['GET'])
    async def agent_check_kb():
        kb_id = request.args.get('id', '')
        if not kb_id:
            return jsonify({"error": "No knowledge base ID provided"}), 400
            
        if not agent_system:
            return jsonify({"error": "Agent system not available"}), 500
            
        try:
            result = await agent_system.check_kb_exists(kb_id)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error checking knowledge base existence: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

async def api_agent_process_message(app):
    """API endpoint to process a message with the agent system"""
    @app.route('/api/agent/process_message', methods=['POST'])
    async def agent_process_message():
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        message = data.get('message', '').strip()
        chat_id = data.get('chat_id')
        kb_id = data.get('kb_id')
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
            
        if not agent_system:
            return jsonify({"error": "Agent system not available"}), 500
            
        try:
            result = await agent_system.process_user_message(message, chat_id, kb_id)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error processing message with agent system: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

async def api_agent_get_best_template(app):
    """API endpoint to get the best template for a query"""
    @app.route('/api/agent/get_best_template', methods=['POST'])
    async def agent_get_best_template():
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        query = data.get('query', '').strip()
        required_variables = data.get('required_variables', [])
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
            
        if not agent_system:
            return jsonify({"error": "Agent system not available"}), 500
            
        try:
            result = await agent_system.get_best_template(query, required_variables)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error getting best template: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

async def api_agent_get_best_kb(app):
    """API endpoint to get the best knowledge base for a query"""
    @app.route('/api/agent/get_best_kb', methods=['POST'])
    async def agent_get_best_kb():
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        query = data.get('query', '').strip()
        topic = data.get('topic')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
            
        if not agent_system:
            return jsonify({"error": "Agent system not available"}), 500
            
        try:
            result = await agent_system.get_best_kb(query, topic)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error getting best knowledge base: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

async def api_agent_get_kb_file_filter(app):
    """API endpoint to get file filter recommendations for a knowledge base"""
    @app.route('/api/agent/get_kb_file_filter', methods=['POST'])
    async def agent_get_kb_file_filter():
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        kb_id = data.get('kb_id', '').strip()
        query = data.get('query', '').strip()
        
        if not kb_id:
            return jsonify({"error": "No knowledge base ID provided"}), 400
            
        if not query:
            return jsonify({"error": "No query provided"}), 400
            
        if not agent_system:
            return jsonify({"error": "Agent system not available"}), 500
            
        try:
            result = await agent_system.get_kb_file_filter(kb_id, query)
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error getting KB file filter: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

async def register_agent_routes(app):
    """Register all agent-related routes"""
    await api_agent_check_template(app)
    await api_agent_check_kb(app)
    await api_agent_process_message(app)
    await api_agent_get_best_template(app)
    await api_agent_get_best_kb(app)
    await api_agent_get_kb_file_filter(app)
