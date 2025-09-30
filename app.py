import argparse
import os
import time
from dotenv import load_dotenv
import faiss
import pandas as pd

from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from providers.mock import MockProvider
from rag.retrieve import retrieve
from rag.prompts import build_user_prompt, get_system_prompt


def _prompt_provider_choice() -> str:
    """Pregunta al usuario qué proveedor desea usar: 'chatgpt' | 'deepseek' | 'mock' | 'compare'."""
    print("\nElige una opción:")
    print("  [1] ChatGPT (OpenRouter/OpenAI)")
    print("  [2] DeepSeek")
    print("  [3] Mock (pruebas sin costo)")
    print("  [4] Compare (ChatGPT vs DeepSeek)")
    while True:
        choice = input("Elige proveedor: ").strip().lower()
        if choice in ("1", "chatgpt", "gpt", "openai"):
            return "chatgpt"
        if choice in ("2", "deepseek", "ds"):
            return "deepseek"
        if choice in ("3", "mock", "m"):
            return "mock"
        if choice in ("4", "compare", "cmp"):
            return "compare"
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
    parser.add_argument('--gold', default='eval/gold_set.jsonl')
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

    # Si eligen 'compare' en el menú, pedir la consulta y comparar antes de entrar al chat
    if provider_key == 'compare':
        comp_q = input("Consulta para comparar (se enviará a ChatGPT y DeepSeek): ").strip()
        if comp_q:
            # Recuperación de contexto una sola vez
            try:
                t_retr0 = time.time()
                retrieved_chunks = retrieve(comp_q, index, chunks_df, args.k)
                t_retr = time.time() - t_retr0
            except FileNotFoundError:
                retrieved_chunks = []
                t_retr = 0.0

            context_docs = []
            for chunk in retrieved_chunks:
                context_docs.append({
                    'content': getattr(chunk, 'content', ''),
                    'source': getattr(chunk, 'source', ''),
                    'page': getattr(chunk, 'page', None),
                    'score': getattr(chunk, 'score', None)
                })

            user_prompt = build_user_prompt(comp_q, context_docs)
            system_prompt = get_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            comparisons = []
            for key in ('deepseek', 'chatgpt'):
                prov = _instantiate_provider(key)
                start = time.time()
                try:
                    ans = prov.chat(messages)
                except Exception as e:
                    ans = f"[Error proveedor] {e}"
                elapsed = time.time() - start
                comparisons.append({
                    'label': key.upper(),
                    'model': prov.name,
                    'time': elapsed,
                    'answer': ans,
                })

            fastest = min(comparisons, key=lambda x: x['time'])
            slowest = max(comparisons, key=lambda x: x['time'])

            print("\n" + "="*80)
            print("🔍 COMPARACIÓN DE RESPUESTAS")
            print(f"Pregunta: {comp_q}")
            print("="*80 + "\n")
            for comp in comparisons:
                print(f"🤖 {comp['label']}")
                print(f"Modelo: {comp['model']}")
                print(f"Tiempo: {comp['time']:.2f}s")
                print("-"*40)
                print(comp['answer'])
                print("-"*40 + "\n")
            print("📊 ESTADÍSTICAS:")
            print(f"Más rápido: {fastest['time']:.2f}s ({fastest['model']})")
            print(f"Más lento: {slowest['time']:.2f}s ({slowest['model']})")
            print("(Retrieve común): {:.2f}s".format(t_retr))
            print()

        # Después de comparar, pedir proveedor para el chat
        provider_key = _prompt_provider_choice()

    provider = _instantiate_provider(provider_key)
    print(f"Proveedor seleccionado: {provider.name}")

    if args.batch:
        # Cargar evaluator solo si es necesario 
        from eval.quality_evaluator import QualityEvaluator
        evaluator = QualityEvaluator(gold_set_path=args.gold, k=args.k)
        evaluator.rag_engine.set_provider(provider)
        metrics = evaluator.run_and_save(provider, provider.name)
        print("Resumen evaluación:", metrics)
        return

    # Modo interactivo
    print("Escribe 'exit' para salir. Comandos: /prov para cambiar proveedor | /compare para comparar | 4 = comparar")
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

        # Comando de comparación rápida entre DeepSeek y ChatGPT
        ql = query.strip().lower()
        if ql.startswith('/compare') or ql == 'compare' or ql == '4':
            if ql in ('/compare', 'compare', '4'):
                comp_q = input("Consulta para comparar (se enviará a ChatGPT y DeepSeek): ").strip()
            else:
                parts = query.split(' ', 1)
                comp_q = parts[1].strip() if len(parts) > 1 else ''
            if not comp_q:
                print("Uso: /compare <pregunta>")
                continue

            # Recuperación de contexto una sola vez
            try:
                t_retr0 = time.time()
                retrieved_chunks = retrieve(comp_q, index, chunks_df, args.k)
                t_retr = time.time() - t_retr0
            except FileNotFoundError:
                retrieved_chunks = []
                t_retr = 0.0

            context_docs = []
            for chunk in retrieved_chunks:
                context_docs.append({
                    'content': getattr(chunk, 'content', ''),
                    'source': getattr(chunk, 'source', ''),
                    'page': getattr(chunk, 'page', None),
                    'score': getattr(chunk, 'score', None)
                })

            user_prompt = build_user_prompt(comp_q, context_docs)
            system_prompt = get_system_prompt()
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Ejecutar con DeepSeek y ChatGPT 
            comparisons = []
            for key in ('deepseek', 'chatgpt'):
                prov = _instantiate_provider(key)
                start = time.time()
                try:
                    ans = prov.chat(messages)
                except Exception as e:
                    ans = f"[Error proveedor] {e}"
                elapsed = time.time() - start
                comparisons.append({
                    'label': key.upper(),
                    'model': prov.name,
                    'time': elapsed,
                    'answer': ans,
                })

            # Determinar más rápido / más lento
            fastest = min(comparisons, key=lambda x: x['time'])
            slowest = max(comparisons, key=lambda x: x['time'])

            # Render
            print("\n" + "="*80)
            print("🔍 COMPARACIÓN DE RESPUESTAS")
            print(f"Pregunta: {comp_q}")
            print("="*80 + "\n")

            for comp in comparisons:
                print(f"🤖 {comp['label']}")
                print(f"Modelo: {comp['model']}")
                print(f"Tiempo: {comp['time']:.2f}s")
                print("-"*40)
                print(comp['answer'])
                print("-"*40 + "\n")

            print("📊 ESTADÍSTICAS:")
            print(f"Más rápido: {fastest['time']:.2f}s ({fastest['model']})")
            print(f"Más lento: {slowest['time']:.2f}s ({slowest['model']})")
            print("(Retrieve común): {:.2f}s".format(t_retr))
            print()
            continue

        # Recuperación de contexto 
        t_start = time.time()
        try:
            t_retr0 = time.time()
            retrieved_chunks = retrieve(query, index, chunks_df, args.k)
            t_retr = time.time() - t_retr0
        except FileNotFoundError as e:
            print(f"[RAG deshabilitado] {e}")
            retrieved_chunks = []
            t_retr = 0.0
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
        # Llamada al proveedor y medición de latencia
        try:
            t_chat0 = time.time()
            response = provider.chat(messages)
            t_chat = time.time() - t_chat0
        except Exception as e:
            response = f"[Error proveedor] {e}"
            t_chat = 0.0
        print(f"\nRespuesta:\n{response}\n")

        # Stats mínimas: latencias, tokens aprox y costo estimado
        def _approx_tokens(text: str) -> int:
            return max(1, len(text) // 4)

        tokens_in = _approx_tokens(str(messages))
        tokens_out = _approx_tokens(response)
        try:
            cost_est = provider.estimate_cost(tokens_in, tokens_out)
        except Exception:
            cost_est = 0.0
        t_total = time.time() - t_start

        print("[Stats] ")
        print(f"  Proveedor: {provider.name}")
        print(f"  k: {args.k} | Chunks recuperados: {len(retrieved_chunks)}")
        print(f"  Latencia retrieve: {t_retr:.3f}s | Latencia chat: {t_chat:.3f}s | Total: {t_total:.3f}s")
        print(f"  Tokens aprox IN: {tokens_in} | OUT: {tokens_out} | Costo estimado: ${cost_est:.6f}")

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