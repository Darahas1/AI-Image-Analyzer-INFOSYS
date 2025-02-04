"""
Centralized configuration for AI services
"""
from openai import OpenAI
import os

def configure_ai():
    """Configure AI services with appropriate API keys and settings"""
    pass

def get_openai_client():
    """Initialize and return OpenAI client"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    return OpenAI(api_key=api_key)

# Standard model configurations
GPT_CONFIG = {
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 150
}

# Response formatting helpers
def format_success_response(data):
    """Format successful response"""
    return {
        'success': True,
        'data': data
    }

def format_error_response(error_message, error_code):
    """Format error response"""
    return {
        'success': False,
        'error': error_message,
        'error_code': error_code
    } 