import os
import time
from typing import Dict, List, Any
from openai import OpenAI
from .base import BaseProvider


class ChatGPTProvider(BaseProvider):
    """Proveedor ChatGPT usando la API de OpenAI."""

    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},  # por 1K tokens
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
    }

    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-4.1-mini"):
        # Permite configurar API key desde variables de entorno
        api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.api_key = api_key or ""
        self.model = model

        # Configurar cliente 
        self.client = None
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
            )

    @property
    def name(self) -> str:
        return f"ChatGPT ({self.model})"

    def chat_detailed(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Envía solicitud de chat completion a OpenAI y retorna metadatos."""
        start_time = time.time()

        try:
            if not self.client:
                # Modo degradado: sin API key devuelve respuesta estática útil para pruebas
                latency = self._measure_latency(start_time)
                dummy = "[ChatGPT deshabilitado: falta API key]"
                tokens_in = max(1, len(str(messages)) // 4)
                tokens_out = len(dummy) // 4
                return {
                    "response": dummy,
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_out,
                    "total_tokens": tokens_in + tokens_out,
                    "latency": latency,
                    "cost": 0.0,
                    "model": self.model,
                }

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.2),
                max_tokens=kwargs.get("max_tokens", 1500),
            )

            latency = self._measure_latency(start_time)

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = response.usage.total_tokens if response.usage else 0
            cost = self.estimate_cost(input_tokens, output_tokens)

            return {
                "response": response.choices[0].message.content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency": latency,
                "cost": cost,
                "model": self.model,
            }

        except Exception as e:
            return {
                "error": str(e),
                "latency": self._measure_latency(start_time),
                "cost": 0.0,
            }

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estima el costo basado en los precios actuales de OpenAI."""
        if self.model in self.PRICING:
            prices = self.PRICING[self.model]
            input_cost = (input_tokens / 1000) * prices["input"]
            output_cost = (output_tokens / 1000) * prices["output"]
            return input_cost + output_cost
        return 0.0

    # Implementación de BaseProvider que retorna solo el texto
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        result = self.chat_detailed(messages, **kwargs)
        if "error" in result:
            raise RuntimeError(result["error"])  
        return result.get("response", "")
