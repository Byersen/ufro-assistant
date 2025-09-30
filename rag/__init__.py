# RAG package
"""
Sistema de Recuperación Aumentada por Generación (RAG)
para el asistente de la Universidad de La Frontera.
"""

from .retrieve import Retriever
from .embedding_system import EmbeddingSystem
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .data_models import DocumentChunk

# Importaciones opcionales (no necesarias para construir índices, p. ej. python -m rag.embed)
try:
    from .rag_engine import RAGEngine, RAGResponse, ask_rag_question  # type: ignore
except Exception:
    RAGEngine = None  # type: ignore
    RAGResponse = None  # type: ignore
    ask_rag_question = None  # type: ignore

__all__ = [
    'Retriever', 
    'EmbeddingSystem', 
    'SYSTEM_PROMPT', 
    'build_user_prompt',
    'RAGEngine',
    'RAGResponse', 
    'ask_rag_question',
    'DocumentChunk'
]