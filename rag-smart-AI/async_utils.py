"""
Utilities for handling asyncio in Flask
"""

import asyncio
import logging
from functools import wraps
from flask import current_app

logger = logging.getLogger(__name__)

def setup_async_for_flask():
    """
    Set up asyncio for Flask by creating a new event loop if needed.
    This should be called at the start of the application.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            logger.info("Event loop was closed, creating a new one")
            asyncio.set_event_loop(asyncio.new_event_loop())
    except RuntimeError:
        logger.info("No event loop found, creating a new one")
        asyncio.set_event_loop(asyncio.new_event_loop())

def run_async(func):
    """
    Decorator to run an async function in a Flask route.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(func(*args, **kwargs))

    return wrapper

def flask_async_route(route_function):
    """
    Decorator for Flask routes that need to run async functions.
    This ensures the route has access to an event loop.
    """
    @wraps(route_function)
    def wrapper(*args, **kwargs):
        # Ensure we have an event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                asyncio.set_event_loop(asyncio.new_event_loop())
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # If the route is async, run it in the event loop
        if asyncio.iscoroutinefunction(route_function):
            return asyncio.get_event_loop().run_until_complete(route_function(*args, **kwargs))
        else:
            # If it's not async, just call it normally
            return route_function(*args, **kwargs)

    return wrapper
