from typing import List, Dict, Any
from .base import BaseProvider

class MockProvider(BaseProvider):
    """
    Proveedor Mock que simula respuestas para pruebas
    """

    def __init__(self):
        self.model = "mock-model"

    @property
    def name(self) -> str:
        return "Mock"

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        Simula una respuesta de chat basada en la consulta
        """
        # Extraer la consulta del ultimo mensaje
        user_message = ""
        for message in messages:
            if message.get("role") == "user":
                user_message = message.get("content", "")
        
        # Generar respuesta mock basada en palabras clave
        query_lower = user_message.lower()
        
        if "matrícula" in query_lower or "matricula" in query_lower:
            return '''Según el Reglamento de Admisión para carreras de Pregrado, "la matrícula es el acto académico mediante el cual el estudiante se incorpora oficialmente a la Universidad y a una carrera específica".

El proceso de matrícula incluye:
- Presentación de documentos requeridos
- Pago de aranceles correspondientes  
- Inscripción de asignaturas según plan de estudios

Referencias:
[Reglamento-de-Admision-para-carreras-de-Pregrado, p.15]'''

        elif "nota" in query_lower or "calificación" in query_lower:
            return '''De acuerdo al Reglamento de Régimen de Estudios 2023, "la escala de calificaciones va de 1.0 a 7.0, siendo la nota mínima de aprobación 4.0".

El sistema establece que:
- Nota máxima: 7.0
- Nota mínima de aprobación: 4.0
- Las calificaciones se expresan con un decimal

Referencias:
[Reglamento-de-Regimen-de-Estudios-2023, p.23]'''

        elif "arancel" in query_lower or "pago" in query_lower:
            return '''Según el Reglamento de Obligaciones Financieras, "los aranceles y derechos universitarios deben cancelarse en los plazos establecidos por la Universidad".

Las obligaciones financieras incluyen:
- Arancel anual de la carrera
- Derechos de matrícula  
- Otros cobros según corresponda

Referencias:
[Reglamento-de-Obligaciones-Financieras, p.8]'''

        elif "título" in query_lower or "titulación" in query_lower:
            return '''El Reglamento de Actividad de Titulación establece que "para obtener el título profesional, el estudiante debe completar satisfactoriamente una actividad de titulación".

Los requisitos incluyen:
- Haber aprobado todas las asignaturas del plan de estudios
- Completar la actividad de titulación según modalidad elegida
- Cumplir con requisitos administrativos

Referencias:
[Reglamento-Actividad-de-Titulacion, p.12]'''

        else:
            return '''Basándome en la normativa disponible de la Universidad de La Frontera, no encontré información específica que responda exactamente a su consulta. 

Le recomiendo:
1. Revisar los reglamentos específicos según su área de interés
2. Contactar directamente con la oficina correspondiente
3. Consultar el sitio web oficial de la UFRO

Referencias:
[Documentos-varios-UFRO, p.N/A]'''

    # Stats simbolicos para futuros tests
    def _count_tokens_approximate(self, text: str) -> int:
        """Estimación aproximada de tokens"""
        return len(text.split()) * 1.3 

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Costo simulado para pruebas"""
        return 0.0001  

    def test_connection(self) -> bool:
        """Siempre disponible para pruebas"""
        return True