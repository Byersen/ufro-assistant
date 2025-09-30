from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from .data_models import DocumentChunk
from .embedding_system import EmbeddingSystem


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if url:
        return QdrantClient(url=url, api_key=api_key)
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return QdrantClient(host=host, port=port, api_key=api_key)


def ensure_collection(client: QdrantClient, collection: str, vector_size: int) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: List[DocumentChunk],
    embeddings: np.ndarray,
) -> None:
    # Qdrant Point ID debe ser int/uuid; usamos ints y guardamos chunk_id real en payload
    points = []
    for idx, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        payload = {
            "doc_id": chunk.doc_id,
            "title": chunk.title,
            "content": chunk.content,
            "page": chunk.page,
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "url": chunk.url,
            "vigencia": chunk.vigencia,
        }
        points.append(PointStruct(id=idx, vector=vec.tolist(), payload=payload))

    client.upsert(collection_name=collection, points=points)


class QdrantRetriever:
    """Retriever simple contra Qdrant."""

    def __init__(self, collection: Optional[str] = None, model_name: str = None):
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "ufro_chunks")
        self.client = get_qdrant_client()
        self.embedding_system = EmbeddingSystem(model_name=model_name or os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))

    def search(self, query: str, k: int = 4) -> List[DocumentChunk]:
        qvec = self.embedding_system.embed_text(query)[0]
        res = self.client.search(
            collection_name=self.collection,
            query_vector=qvec.tolist(),
            limit=k,
            with_payload=True,
        )
        results: List[DocumentChunk] = []
        for p in res:
            pl = p.payload or {}
            chunk = DocumentChunk(
                content=pl.get("content", ""),
                source=str(pl.get("source", pl.get("doc_id", ""))),
                page=int(pl.get("page", 0)),
                chunk_id=str(pl.get("chunk_id", "")),
                doc_id=str(pl.get("doc_id", "")),
                title=str(pl.get("title", "")),
                url=str(pl.get("url", "")),
                vigencia=str(pl.get("vigencia", "")),
            )
            # Anotar score (mayor = más similar con cosine)
            setattr(chunk, "score", float(p.score))
            results.append(chunk)
        return results
