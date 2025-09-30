# rag/embed.py
import os
import pandas as pd
from dotenv import load_dotenv
from rag.embedding_system import EmbeddingSystem
from rag.data_models import DocumentChunk

load_dotenv()

PROCESSED_FILE = "data/processed/chunks.parquet"
INDEX_FILE = "data/index.faiss"
CHUNKS_WITH_EMBEDDINGS = "data/processed/chunks_with_embeddings.parquet"

def load_chunks_from_parquet(file_path: str):
    """Carga chunks desde archivo parquet y los convierte a DocumentChunk."""
    df = pd.read_parquet(file_path)
    chunks = []
    
    # Detectar columnas disponibles y mapear
    cols = set(df.columns)
    for i, row in df.iterrows():
        # Soportar esquemas: (doc,text,chunk_id) o (doc_id,content,chunk_id,page,...)
        doc_id = row.get('doc_id', row.get('doc', f'doc_{i}'))
        title = row.get('title', row.get('doc', str(doc_id)))
        content = row.get('content', row.get('text', ''))
        page = int(row.get('page', 1))
        chunk_id = str(row.get('chunk_id', i))
        url = row.get('url', '')
        vigencia = row.get('vigencia', '')

        chunk = DocumentChunk(
            content=content,
            source=str(doc_id),
            page=page,
            chunk_id=chunk_id,
            doc_id=str(doc_id),
            title=str(title),
            url=str(url) if pd.notna(url) else "",
            vigencia=str(vigencia) if pd.notna(vigencia) else "",
        )
        chunks.append(chunk)
    
    return chunks

def build_index():
    """Construye índice FAISS usando EmbeddingSystem."""
    try:
        print(f"Cargando chunks desde {PROCESSED_FILE}...")
        chunks = load_chunks_from_parquet(PROCESSED_FILE)
        print(f"Cargados {len(chunks)} chunks")

        # Inicializa sistema de embeddings
        model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        embedding_system = EmbeddingSystem(model_name=model_name)

        # Construye y guarda índice
        embedding_system.build_and_save_index(
            chunks=chunks,
            index_path=INDEX_FILE,
            chunks_path=CHUNKS_WITH_EMBEDDINGS
        )

        print(f"✅ Índice FAISS guardado en {INDEX_FILE}")
        print(f"✅ Chunks con embeddings guardados en {CHUNKS_WITH_EMBEDDINGS}")

    except FileNotFoundError as e:
        print(f"❌ Error: No se pudo encontrar el archivo {PROCESSED_FILE}")
        print(f"   Asegúrate de ejecutar el procesamiento de documentos primero")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        raise

if __name__ == "__main__":
    build_index()
