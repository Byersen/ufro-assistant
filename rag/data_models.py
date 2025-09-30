import os
import hashlib
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class DocumentChunk:
    """
    Representa un fragmento de texto de un documento para el sistema RAG.
    Versión unificada que combina funcionalidad de DocumentChunk y ChunkRecord.
    """
    # Contenido principal
    content: str                 
    source: str                  
    
    # Metadatos del documento
    page: int = 0               
    chunk_id: str = ""         
    doc_id: str = ""         
    title: str = ""           
    
    # Metadatos adicionales
    url: str = ""         
    vigencia: str = ""          
    
    # Metadatos técnicos
    chunk_size: int = 0       
    overlap: int = 0          
    
    def __post_init__(self):
        """Normaliza y completa metadatos cuando faltan (pensado para PDFs en directorios)."""
        # Normalizar source si es ruta
        if self.source:
            self.source = os.path.basename(self.source).strip()

        # doc_id por defecto toma el source normalizado
        if not self.doc_id and self.source:
            self.doc_id = self.source

        # Título legible si falta
        if not self.title and self.source:
            name = self.source
            if name.lower().endswith('.pdf'):
                name = name[:-4]
            self.title = name.replace('_', ' ').replace('-', ' ').strip().capitalize()

        # Tamaño del chunk si falta
        if not self.chunk_size:
            self.chunk_size = len(self.content or "")

        # chunk_id determinístico si falta
        if not self.chunk_id:
            h = hashlib.md5()
            h.update((str(self.doc_id) + '|' + str(self.page) + '|' + (self.content or '')).encode('utf-8'))
            self.chunk_id = f"chunk-{h.hexdigest()[:16]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el chunk a diccionario para serialización."""
        return {
            'content': self.content,
            'source': self.source,
            'page': self.page,
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'title': self.title,
            'url': self.url,
            'vigencia': self.vigencia,
            'chunk_size': self.chunk_size,
            'overlap': self.overlap
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentChunk':
        """Crea un DocumentChunk desde un diccionario."""
        return cls(
            content=data.get('content', data.get('text', '')),
            source=data.get('source', ''),
            page=data.get('page', 0),
            chunk_id=data.get('chunk_id', ''),
            doc_id=data.get('doc_id', ''),
            title=data.get('title', ''),
            url=data.get('url', ''),
            vigencia=data.get('vigencia', ''),
            chunk_size=data.get('chunk_size', 0),
            overlap=data.get('overlap', 0)
        )
    
    def get_display_name(self) -> str:
        """Retorna un nombre para mostrar en las respuestas."""
        if self.title and self.title != self.source:
            return f"{self.title} (pág. {self.page})"
        return f"{self.source} (pág. {self.page})"

    @property
    def page_number(self) -> int:
        """Compatibilidad: algunos módulos consultan page_number; mapea a page."""
        return self.page

    @classmethod
    def from_file_fragment(
        cls,
        file_path: str,
        page: int,
        content: str,
        *,
        title: Optional[str] = None,
        url: str = "",
        vigencia: str = "",
        doc_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> 'DocumentChunk':
        """Crea un chunk a partir de un archivo local (PDF/TXT) y una página.

        - source se deriva del nombre de archivo
        - doc_id por defecto toma el source
        - chunk_id se calcula de forma estable si no se entrega
        """
        source = os.path.basename(file_path).strip()
        did = doc_id or source
        if not chunk_id:
            h = hashlib.md5()
            h.update((str(did) + '|' + str(page) + '|' + (content or '')).encode('utf-8'))
            chunk_id = f"chunk-{h.hexdigest()[:16]}"
        ttl = title or (source[:-4] if source.lower().endswith('.pdf') else source)
        ttl = ttl.replace('_', ' ').replace('-', ' ').strip().capitalize()
        return cls(
            content=content or "",
            source=source,
            page=int(page or 0),
            chunk_id=chunk_id,
            doc_id=did,
            title=ttl,
            url=url or "",
            vigencia=vigencia or "",
        )
