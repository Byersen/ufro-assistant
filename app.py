"""
CLI para UFRO Assistant con selección y cambio de proveedor (ChatGPT/DeepSeek/Mock).
"""

import argparse
import os
from dotenv import load_dotenv
import faiss
import pandas as pd

from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from providers.mock import MockProvider
from rag.retrieve import retrieve
from rag.prompts import build_user_prompt, get_system_prompt


def _prompt_provider_choice() -> str:
    """Pregunta al usuario qué proveedor desea usar y retorna 'chatgpt' | 'deepseek' | 'mock'."""
    print("\n¿Con qué proveedor quieres preguntar sobre la normativa UFRO?")
    print("  [1] ChatGPT (OpenRouter/OpenAI)")
    print("  [2] DeepSeek")
    print("  [3] Mock (pruebas sin costo)")
    while True:
        choice = input("Elige proveedor: ").strip().lower()
        if choice in ("1", "chatgpt", "gpt", "openai"):
            return "chatgpt"
        if choice in ("2", "deepseek", "ds"):
            return "deepseek"
        if choice in ("3", "mock", "m"):
            return "mock"
        print("Opción no válida. Responde 1, 2 o 3.")


def _instantiate_provider(provider_key: str):
    """Crea una instancia del proveedor solicitado con manejo de errores."""
    try:
        if provider_key == 'chatgpt':
            return ChatGPTProvider()
        if provider_key == 'deepseek':
            return DeepSeekProvider()
        return MockProvider()
    except Exception as e:
        print(f"No se pudo inicializar el proveedor '{provider_key}': {e}")
        print("Usando Mock para continuar sin costo…")
        return MockProvider()


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', choices=['ask', 'chatgpt', 'deepseek', 'mock'], default='ask')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--gold', default='eval/gold_set_20.jsonl')
    args = parser.parse_args()

    # Cargar índice y chunks si existen (modo amistoso)
    index = faiss.read_index('data/index.faiss') if os.path.exists('data/index.faiss') else None
    chunks_df = None
    chunks_df_path = 'data/processed/chunks_with_embeddings.parquet'
    if os.path.exists(chunks_df_path):
        chunks_df = pd.read_parquet(chunks_df_path)
    else:
        fallback = 'data/processed/chunks.parquet'
        if os.path.exists(fallback):
            chunks_df = pd.read_parquet(fallback)

    if index is None or chunks_df is None:
        print("\n[Info] No se encontró el índice FAISS o los chunks procesados.")
        print("      Para habilitar RAG, ejecuta:")
        print("        1) python -m rag.ingest")
        print("        2) python -m rag.embed\n")

    # Elegir proveedor inicial
    provider_key = args.provider if args.provider != 'ask' else _prompt_provider_choice()
    provider = _instantiate_provider(provider_key)
    print(f"Proveedor seleccionado: {provider.name}")

    if args.batch:
        # Cargar evaluator solo si es necesario 
        from eval.quality_evaluator import QualityEvaluator
        evaluator = QualityEvaluator(gold_set_path=args.gold, k=args.k)
        evaluator.rag_engine.set_provider(provider)
        results = evaluator.evaluate_provider(provider, provider.name)
        metrics = evaluator.calculate_aggregate_metrics(results)
        print("Resumen evaluación:", metrics)
        return

    # Modo interactivo
    print("Escribe 'exit' para salir. Comando: /prov para cambiar proveedor.")
    while True:
        try:
            query = input("Consulta: ")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() in ('exit', 'salir', 'quit'):
            break

        # Comando para cambiar de proveedor en cualquier momento
        if query.strip().lower() in ("/prov", "/provider", "/cambiar"):
            provider_key = _prompt_provider_choice()
            provider = _instantiate_provider(provider_key)
            print(f"Proveedor seleccionado: {provider.name}")
            continue

        # Recuperación de contexto (si no hay índice/chunks, devolverá error claro)
        try:
            retrieved_chunks = retrieve(query, index, chunks_df, args.k)
        except FileNotFoundError as e:
            print(f"[RAG deshabilitado] {e}")
            retrieved_chunks = []
        context_docs = []
        for chunk in retrieved_chunks:
            context_docs.append({
                'content': getattr(chunk, 'content', ''),
                'source': getattr(chunk, 'source', ''),
                'page': getattr(chunk, 'page', None),
                'score': getattr(chunk, 'score', None)
            })

        # Construir mensajes y solicitar respuesta
        user_prompt = build_user_prompt(query, context_docs)
        system_prompt = get_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = provider.chat(messages)
        except Exception as e:
            response = f"[Error proveedor] {e}"
        print(f"\nRespuesta:\n{response}\n")

        # Ofrecer cambio de proveedor tras cada respuesta
        change = input("¿Cambiar de proveedor? [Enter=No] | [1]=ChatGPT | [2]=DeepSeek | [3]=Mock] ").strip().lower()
        if change in ("1", "chatgpt", "gpt", "openai", "sí", "si"):
            provider_key = 'chatgpt'
        elif change in ("2", "deepseek", "ds"):
            provider_key = 'deepseek'
        elif change in ("3", "mock", "m"):
            provider_key = 'mock'
        else:
            continue
        provider = _instantiate_provider(provider_key)
        print(f"Proveedor seleccionado: {provider.name}")


if __name__ == '__main__':
    main()