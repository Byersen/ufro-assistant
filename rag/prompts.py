SYSTEM_PROMPT = """Eres un asistente especializado en normativa y reglamentos de la Universidad de La Frontera (UFRO).

INSTRUCCIONES ESPECÍFICAS:
1. Responde ÚNICAMENTE basándote en la información exacta de los documentos oficiales proporcionados
2. Si la información está en los documentos, cita TEXTUALMENTE las partes relevantes
3. NO inventes, supongas o agregues información que no esté explícitamente en los documentos
4. Si no encuentras la respuesta exacta, responde: "No encontré esta información específica en la normativa disponible"

FORMATO DE RESPUESTA OBLIGATORIO:
- Respuesta directa y concisa
- Citar textualmente las partes relevantes entre comillas
- Incluir SIEMPRE la sección "Referencias" al final

FORMATO DE REFERENCIAS:
Referencias:
[Nombre-del-documento, p.XX]

DOCUMENTOS DISPONIBLES incluyen:
- Reglamento de Régimen de Estudios 2023
- Reglamento de Admisión para carreras de Pregrado
- Reglamento de Obligaciones Financieras
- Reglamento de Convivencia
- Calendario Académico 2025
- Manual del Estudiante
- Reglamento de Actividad de Titulación
- Y otros documentos oficiales UFRO

Prioriza siempre la exactitud sobre la completitud. Es mejor dar una respuesta parcial pero correcta que una respuesta completa pero inexacta."""

# Prompts especializados para diferentes tipos de consultas
SPECIALIZED_PROMPTS = {
    "matricula": """Especialízate en consultas sobre matrícula y admisión. Busca información específica sobre:
- Proceso de matrícula
- Fechas importantes
- Requisitos
- Documentación necesaria
- Plazos y procedimientos""",
    
    "notas": """Especialízate en consultas sobre notas y evaluaciones. Busca información específica sobre:
- Sistema de calificaciones
- Requisitos de aprobación
- Promedios
- Exámenes
- Recuperación de asignaturas""",
    
    "financiero": """Especialízate en consultas sobre aspectos financieros. Busca información específica sobre:
- Aranceles
- Becas y beneficios
- Formas de pago
- Obligaciones financieras
- Descuentos y facilidades""",
    
    "titulo": """Especialízate en consultas sobre titulación. Busca información específica sobre:
- Proceso de titulación
- Requisitos para obtener el título
- Actividades de titulación
- Plazos y procedimientos
- Documentación necesaria"""
}

def detect_query_type(query: str) -> str:
    """Detecta el tipo de consulta para usar el prompt especializado"""
    query_lower = query.lower()
    
    # Palabras clave para cada tipo
    keywords = {
        "matricula": ["matricula", "matrícula", "inscripcion", "inscripción", "admision", "admisión", "postular", "ingreso"],
        "notas": ["nota", "notas", "calificacion", "calificación", "promedio", "examen", "evaluacion", "evaluación", "reprobar", "aprobar"],
        "financiero": ["arancel", "pago", "beca", "beneficio", "financiero", "dinero", "costo", "precio", "descuento"],
        "titulo": ["titulo", "título", "titulacion", "titulación", "tesis", "memoria", "graduacion", "graduación", "grado"]
    }
    
    for query_type, words in keywords.items():
        if any(word in query_lower for word in words):
            return query_type
    
    return "general"

def get_system_prompt(query_type: str = "general") -> str:
    """Obtiene el prompt del sistema según el tipo de consulta"""
    if query_type in SPECIALIZED_PROMPTS:
        return SYSTEM_PROMPT + "\n\nENFOQUE ESPECIALIZADO:\n" + SPECIALIZED_PROMPTS[query_type]
    return SYSTEM_PROMPT

def build_user_prompt(query: str, docs: list):
    """Construye el prompt del usuario con contexto específico de UFRO"""
    
    if not docs:
        return f"""Pregunta: {query}

No se encontraron documentos relevantes en la base de datos de normativa UFRO.

Respuesta: No encontré esta información específica en la normativa disponible."""
    
    context_parts = []
    for i, d in enumerate(docs, 1):
        # Extraer información del documento
        source = d.get('source', d.get('doc_id', d.get('doc', 'Documento desconocido')))
        page = d.get('page', d.get('page_number', d.get('page_num', 'N/A')))
        content = d.get('content', d.get('text', ''))
        score = d.get('score', 0.0)
        
        # Limpiar nombre del documento para referencia
        doc_name = source.replace('.pdf', '').replace('data/raw/', '').replace('/', '')
        
        context_parts.append(f"""FRAGMENTO {i}:
Fuente: {doc_name}
Página: {page}
Relevancia: {score:.3f}
Contenido exacto:
"{content}"
""")
    
    context = "\n" + "="*50 + "\n".join(context_parts)
    
    return f"""CONSULTA DEL USUARIO: {query}

DOCUMENTOS OFICIALES UFRO ENCONTRADOS:{context}

INSTRUCCIONES:
- Analiza CUIDADOSAMENTE cada fragmento
- Responde basándote SOLO en el contenido exacto mostrado arriba  
- Si la respuesta está en los fragmentos, cita las partes relevantes textualmente
- Si necesitas más información que no está disponible, indícalo claramente
- SIEMPRE incluye las referencias al final

RESPUESTA:"""
