from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .data_models import DocumentChunk
from .embedding_system import EmbeddingSystem
from .vector_store_qdrant import get_qdrant_client, ensure_collection, upsert_chunks


PROCESSED = Path("data/processed/chunks_with_embeddings.parquet")
FALLBACK = Path("data/processed/chunks.parquet")


def main():
    if PROCESSED.exists():
        df = pd.read_parquet(PROCESSED)
    elif FALLBACK.exists():
        df = pd.read_parquet(FALLBACK)
    else:
        print("[qdrant_upsert] No hay parquet de chunks. Ejecuta 'python -m rag.ingest' primero.")
        return

    chunks = []
    for _, row in df.iterrows():
        chunks.append(DocumentChunk(
            content=row.get('content', ''),
            source=str(row.get('doc_id', '')),
            page=int(row.get('page', 0)),
            chunk_id=str(row.get('chunk_id', '')),
            doc_id=str(row.get('doc_id', '')),
            title=str(row.get('title', '')),
            url=str(row.get('url', '')),
            vigencia=str(row.get('vigencia', '')),
        ))

    model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    embedder = EmbeddingSystem(model_name=model_name)

    texts = [c.content for c in chunks]
    embeddings = embedder.embed_texts(texts)

    client = get_qdrant_client()
    collection = os.getenv("QDRANT_COLLECTION", "ufro_chunks")
    ensure_collection(client, collection, vector_size=embeddings.shape[1])
    upsert_chunks(client, collection, chunks, embeddings)

    print(f"[qdrant_upsert] Upsert de {len(chunks)} chunks a la colección '{collection}' completado.")


if __name__ == "__main__":
    main()
