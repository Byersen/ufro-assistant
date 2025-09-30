from dataclasses import dataclass
from typing import Optional, List, Dict, Any


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
        """Genera automáticamente algunos campos si no se proporcionan."""
        if not self.chunk_id and self.source:
            # Generar ID único basado en fuente y página
            self.chunk_id = f"{self.source}_{self.page}_{hash(self.content[:50]) % 10000}"
        
        if not self.doc_id and self.source:
            self.doc_id = self.source
            
        if not self.title and self.source:
            self.title = self.source.replace('.pdf', '').replace('_', ' ')
            
        if not self.chunk_size:
            self.chunk_size = len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte el chunk a diccionario para serialización."""
        return {
            'content': self.content,  # Cambiado de text a content
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


@dataclass
class ProcessingStats:
    """Estadísticas del procesamiento de documentos."""
    total_documents: int = 0
    total_chunks: int = 0
    total_characters: int = 0
    processing_time: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    def add_error(self, error: str):
        """Agrega un error a la lista."""
        self.errors.append(error)
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna un resumen de las estadísticas."""
        return {
            'documents_processed': self.total_documents,
            'chunks_generated': self.total_chunks,
            'total_characters': self.total_characters,
            'processing_time_seconds': self.processing_time,
            'error_count': len(self.errors),
            'errors': self.errors[:5] if self.errors else []  # Primeros 5 errores
        }


# Alias para compatibilidad con código existente
ChunkRecord = DocumentChunk  

# Constantes útiles
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
MAX_CHUNK_SIZE = 2000
MIN_CHUNK_SIZE = 100