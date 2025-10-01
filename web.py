from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import faiss
import pandas as pd

from providers.chatgpt import ChatGPTProvider
from providers.deepseek import DeepSeekProvider
from providers.mock import MockProvider
from rag.retrieve import retrieve
from rag.prompts import build_user_prompt, get_system_prompt


load_dotenv()
app = Flask(__name__)

# Cache ligero en memoria para acelerar primeras consultas
INDEX_PATH = os.getenv("FAISS_INDEX", "data/index.faiss")
CHUNKS_PATH = os.getenv("CHUNKS_PARQUET", "data/processed/chunks_with_embeddings.parquet")
_INDEX = None
_CHUNKS_DF = None


def _load_rag_cache():
    global _INDEX, _CHUNKS_DF
    if _INDEX is None and os.path.exists(INDEX_PATH):
        _INDEX = faiss.read_index(INDEX_PATH)
    if _CHUNKS_DF is None:
        if os.path.exists(CHUNKS_PATH):
            _CHUNKS_DF = pd.read_parquet(CHUNKS_PATH)
        else:
            # Fallback al archivo sin embeddings
            fallback = "data/processed/chunks.parquet"
            if os.path.exists(fallback):
                _CHUNKS_DF = pd.read_parquet(fallback)


def _instantiate_provider(provider_key: str):
    key = (provider_key or "").strip().lower()
    try:
        if key == "chatgpt":
            return ChatGPTProvider()
        if key == "deepseek":
            return DeepSeekProvider()
        if key == "mock":
            return MockProvider()
    except Exception as e:
        # Si falla el proveedor (p.ej. falta de API key), degradar a Mock
        print(f"[web] No se pudo inicializar '{provider_key}': {e}")
        return MockProvider()
    return MockProvider()


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _format_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for d in docs:
        source = d.get("source") or d.get("doc_id") or d.get("doc") or "Documento"
        page = d.get("page") or d.get("page_number") or "N/A"
        content = d.get("content") or d.get("text") or ""
        score = float(d.get("score") or 0.0)
        out.append({
            "source": str(source),
            "page": page,
            "content": content,
            "score": score,
        })
    return out

def _provider_status() -> Dict[str, Any]:
    """Detecta si hay API keys configuradas para cada proveedor."""
    chatgpt_ok = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
    deepseek_ok = bool(os.getenv("DEEPSEEK_API_KEY"))
    return {
        "chatgpt": chatgpt_ok,
        "deepseek": deepseek_ok,
        "mock": True,
    }

@app.after_request
def add_no_cache_headers(response):
    """Evita que el navegador guarde en caché las páginas/respuestas.

    Esto ayuda a que al volver/recargar no se muestre la última respuesta renderizada.
    """
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/")
def index():
    providers = [
        {"key": "chatgpt", "label": "ChatGPT"},
        {"key": "deepseek", "label": "DeepSeek"},
        {"key": "mock", "label": "Mock (sin costo)"},
        {"key": "compare", "label": "Comparar (ChatGPT vs DeepSeek)"},
    ]
    return render_template("index.html", providers=providers, provider_status=_provider_status(), result=None)


@app.post("/ask")
def ask():
    _load_rag_cache()

    query = (request.form.get("query") or (request.json.get("query") if request.is_json else None) or "").strip()
    provider_key = (request.form.get("provider") or (request.json.get("provider") if request.is_json else None) or "mock").strip().lower()
    k = int((request.form.get("k") or (request.json.get("k") if request.is_json else None) or 5))

    print(f"[web] Incoming ask: provider='{provider_key}', k={k}, len(query)={len(query)}")

    # Recuperación de contexto
    try:
        t_retr0 = time.time()
        retrieved_chunks = retrieve(query, index=_INDEX, chunks_df=_CHUNKS_DF, k=k)
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

    user_prompt = build_user_prompt(query, context_docs)
    system_prompt = get_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    result: Dict[str, Any]
    if provider_key == "compare":
        comps = []
        for key in ("deepseek", "chatgpt"):
            prov = _instantiate_provider(key)
            t0 = time.time()
            try:
                ans = prov.chat(messages)
            except Exception as e:
                ans = f"[Error proveedor] {e}"
            latency = time.time() - t0
            comps.append({
                "label": key.upper(),
                "model": getattr(prov, "model", prov.name),
                "time_sec": latency,
                "answer": ans,
            })
        print("[web] Compare done: deepseek vs chatgpt")
        result = {
            "mode": "compare",
            "question": query,
            "retrieval_sec": t_retr,
            "comparisons": comps,
            "docs": _format_docs(context_docs),
            "provider_status": _provider_status(),
        }
    else:
        prov = _instantiate_provider(provider_key)
        t0 = time.time()
        try:
            answer = prov.chat(messages)
        except Exception as e:
            answer = f"[Error proveedor] {e}"
        chat_sec = time.time() - t0
        tokens_in = _approx_tokens(str(messages))
        tokens_out = _approx_tokens(answer)
        try:
            cost_est = prov.estimate_cost(tokens_in, tokens_out)
        except Exception:
            cost_est = 0.0
        print(f"[web] Provider used: class={prov.__class__.__name__}, model='{getattr(prov, 'model', prov.name)}'")
        result = {
            "mode": "single",
            "question": query,
            "provider": provider_key,
            "model": getattr(prov, "model", prov.name),
            "answer": answer,
            "retrieval_sec": t_retr,
            "chat_sec": chat_sec,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_est": cost_est,
            "docs": _format_docs(context_docs),
            "provider_status": _provider_status(),
        }

    if request.is_json:
        return jsonify(result)

    providers = [
        {"key": "chatgpt", "label": "ChatGPT"},
        {"key": "deepseek", "label": "DeepSeek"},
        {"key": "mock", "label": "Mock (sin costo)"},
        {"key": "compare", "label": "Comparar (ChatGPT vs DeepSeek)"},
    ]
    return render_template("index.html", providers=providers, provider_status=_provider_status(), result=result)


if __name__ == "__main__":
    # Puerto configurable por env var (útil en EC2)
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)
