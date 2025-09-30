"""
Ingesta mínima de documentos:
- Recorre data/raw/
- Extrae texto de PDFs y TXT
- Genera chunks simples por página (PDF) o documento (TXT)
- Guarda data/processed/chunks.parquet

Requisitos: pypdf, pandas, pyarrow
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import List, Dict

import pandas as pd
from pypdf import PdfReader


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CHUNKS_PATH = PROCESSED_DIR / "chunks.parquet"


def _slug_title(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    return name.replace('_', ' ').replace('-', ' ').strip().capitalize()


def _make_chunk_id(doc_id: str, page: int, content: str) -> str:
    h = hashlib.md5()
    h.update((str(doc_id) + '|' + str(page) + '|' + (content or '')).encode('utf-8'))
    return f"chunk-{h.hexdigest()[:16]}"


def ingest() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []

    for path in sorted(RAW_DIR.glob('*')):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        doc_id = path.name
        title = _slug_title(path.name)

        try:
            if ext == ".pdf":
                reader = PdfReader(str(path))
                for i, page in enumerate(reader.pages, start=1):
                    # pypdf extract_text puede retornar None si la página es solo imagen
                    text = page.extract_text() or ""
                    text = text.strip()
                    if not text:
                        continue
                    records.append({
                        'doc_id': doc_id,
                        'title': title,
                        'content': text,
                        'page': i,
                        'chunk_id': _make_chunk_id(doc_id, i, text),
                        'url': '',
                        'vigencia': ''
                    })

            elif ext in (".txt", ".md"):
                text = path.read_text(encoding='utf-8', errors='ignore').strip()
                if text:
                    records.append({
                        'doc_id': doc_id,
                        'title': title,
                        'content': text,
                        'page': 1,
                        'chunk_id': _make_chunk_id(doc_id, 1, text),
                        'url': '',
                        'vigencia': ''
                    })
            else:
                # Ignorar otros tipos de archivo
                continue

        except Exception as e:
            print(f"[ingest] Error procesando {path.name}: {e}")
            continue

    if not records:
        print("[ingest] No se encontraron textos extraíbles en data/raw/")
    else:
        df = pd.DataFrame(records)
        df.to_parquet(CHUNKS_PATH, index=False)
        print(f"[ingest] Guardado {len(df)} chunks en {CHUNKS_PATH}")

    return CHUNKS_PATH


if __name__ == "__main__":
    ingest()
