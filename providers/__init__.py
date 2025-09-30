"""
Proveedores de LLM para UFRO Assistant
"""

from .base import BaseProvider
from .chatgpt import ChatGPTProvider
from .deepseek import DeepSeekProvider
from .mock import MockProvider

__all__ = ['BaseProvider', 'ChatGPTProvider', 'DeepSeekProvider', 'MockProvider']