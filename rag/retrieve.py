import os
import faiss
import pandas as pd
from typing import List
from .embedding_system import EmbeddingSystem
from .data_models import DocumentChunk

class Retriever:
    """Sistema de búsqueda vectorial usando FAISS"""

    def __init__(self, index_path: str = "data/index.faiss",
                 chunks_path: str = "data/processed/chunks_with_embeddings.parquet"):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.embedding_system = EmbeddingSystem()
        self.index = None
        self.chunks: List[DocumentChunk] = []

        self._load_index_and_chunks()

    def _load_index_and_chunks(self):
        """Carga el índice FAISS y los chunks procesados"""
        if not os.path.exists(self.index_path) or not os.path.exists(self.chunks_path):
            raise FileNotFoundError("FAISS index o chunks no encontrados")
        
        self.index = faiss.read_index(self.index_path)
        df = pd.read_parquet(self.chunks_path)
        
        # Mapear columnas del parquet al modelo DocumentChunk
        chunks_data = []
        for _, row in df.iterrows():
            # Soportar esquemas variados
            content = row.get('content', row.get('text', ''))
            doc_id = row.get('doc_id', row.get('doc', ''))
            page = int(row.get('page', 0))
            chunk_id = str(row.get('chunk_id', ''))
            title = row.get('title', str(doc_id))
            url = row.get('url', '')
            vigencia = row.get('vigencia', '')

            chunk_data = {
                'content': content,
                'source': str(doc_id),
                'page': page,
                'chunk_id': chunk_id,
                'doc_id': str(doc_id),
                'title': str(title),
                'url': url if pd.notna(url) else "",
                'vigencia': vigencia if pd.notna(vigencia) else "",
            }
            chunks_data.append(chunk_data)
        
        self.chunks = [DocumentChunk(**chunk_data) for chunk_data in chunks_data]

    def embed_query(self, query: str):
        """Convierte la query en un vector"""
        return self.embedding_system.embed_text(query)

    def search(self, query: str, k: int = 4) -> List[DocumentChunk]:
        """Busca los k documentos más relevantes"""
        query_vec = self.embed_query(query)
        D, I = self.index.search(query_vec, k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            chunk.score = float(score)
            results.append(chunk)
        return results


def retrieve(query: str, index=None, chunks_df: pd.DataFrame | None = None, k: int = 4) -> List[DocumentChunk]:
    """Función de conveniencia para recuperar documentos como lista de chunks.

    Si se provee un índice y un DataFrame de chunks ya cargados, se ignoran las rutas por defecto.
    """
    retriever = Retriever()
    # Nota: Para simplicidad mantenemos carga interna; se podría optimizar para usar index/chunks_df externos.
    return retriever.search(query, k)
