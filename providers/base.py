from abc import ABC, abstractmethod
from typing import List, Dict
import time


class BaseProvider(ABC):
    """Interfaz base para todos los proveedores de LLM.

    Contrato mínimo:
    - chat(messages) -> str: retorna solo el texto de respuesta
    - estimate_cost(input_tokens, output_tokens) -> float
    - name: propiedad legible del proveedor
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        pass

    @abstractmethod
    def estimate_cost(self, input_tokens: int, output_tokens: int = 0) -> float:
        pass

    def _measure_latency(self, start_time: float) -> float:
        return time.time() - start_time

    def _count_tokens_approximate(self, text: str) -> int:
        # Aproximación muy simple para evitar dependencia de tokenizadores
        return max(1, len(text) // 4)
