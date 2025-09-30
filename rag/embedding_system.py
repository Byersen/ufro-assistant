from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pandas as pd
from typing import List
from .data_models import DocumentChunk

class EmbeddingSystem:
    """Generador de embeddings vectoriales"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str):
        """Convierte un texto en vector numpy"""
        vec = self.model.encode([text])
        return np.array(vec, dtype="float32")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embebe una lista de textos a matriz numpy (n, d)."""
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")

    def build_and_save_index(self, chunks: List[DocumentChunk], index_path: str, chunks_path: str):
        """Construye un índice FAISS y guarda los chunks con embeddings."""
        if not chunks:
            raise ValueError("No hay chunks para indexar")

        texts = [c.content for c in chunks]
        embs = self.embed_texts(texts)

        d = embs.shape[1]
        index = faiss.IndexFlatIP(d)
        # Aseguramos que los embeddings estén normalizados para IP ~ coseno
        faiss.normalize_L2(embs)
        index.add(embs)

        faiss.write_index(index, index_path)

        # Guardar chunks con embeddings en parquet
        data = []
        for c in chunks:
            data.append({
                'doc_id': c.doc_id,
                'title': c.title,
                'content': c.content,
                'page': c.page,
                'chunk_id': c.chunk_id,
                'url': c.url,
                'vigencia': c.vigencia
            })
        df = pd.DataFrame(data)
        df.to_parquet(chunks_path, index=False)
