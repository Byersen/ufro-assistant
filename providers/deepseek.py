import os
import time
import requests
from typing import List, Dict, Any
from .base import BaseProvider


class DeepSeekProvider(BaseProvider):
    """
    Proveedor DeepSeek usando API compatible con OpenAI.
    """

    # Modelos soportados y precios (USD por 1K tokens)
    SUPPORTED_MODELS = {
        "deepseek-chat": {
            "name": "DeepSeek Chat", 
            "input": 0.00014, 
            "output": 0.00028
        },
        "deepseek-reasoner": {
            "name": "DeepSeek Reasoner", 
            "input": 0.00055, 
            "output": 0.0022
        },
    }

    def __init__(self, model: str = None):
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        # Endpoint compatible OpenAI
        self.endpoint = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1/chat/completions")
        
        # Configuración por defecto
        self.default_temperature = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
        self.default_max_tokens = int(os.getenv("DEFAULT_MAX_TOKENS", "2000"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "60"))
        
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY no está configurado en el archivo .env")

        # Validar modelo
        if self.model not in self.SUPPORTED_MODELS:
            print(f"⚠  Modelo {self.model} no está en la lista de modelos probados")

    @property
    def name(self) -> str:
        model_name = self.SUPPORTED_MODELS.get(self.model, {}).get("name", self.model)
        return f"deepseek ({model_name})"

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Envía solicitud de chat completion a DeepSeek."""
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.default_temperature),
            "max_tokens": kwargs.get("max_tokens", self.default_max_tokens),
            "stream": False
        }
        
        try:
            response = requests.post(
                self.endpoint,
                headers=headers, 
                json=payload,
                timeout=self.request_timeout
            )
            
            # Manejo especifico de errores HTTP
            if response.status_code == 401:
                raise RuntimeError("Error de autenticación (401): Verifica tu DEEPSEEK_API_KEY")
            elif response.status_code == 400:
                error_data = response.json().get("error", {})
                error_msg = error_data.get("message", response.text)
                raise RuntimeError(f"Error en la solicitud (400): {error_msg}")
            elif response.status_code == 429:
                raise RuntimeError("Rate limit (429): Demasiadas solicitudes a DeepSeek")
            elif response.status_code == 500:
                raise RuntimeError("Error interno del servidor (500) en DeepSeek")
            elif response.status_code == 503:
                raise RuntimeError("Servicio no disponible (503) en DeepSeek")
            
            response.raise_for_status()
            result = response.json()

            if "choices" not in result or not result.get("choices"):
                raise RuntimeError(f"Respuesta vacia de DeepSeek: {result}")
            
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Timeout despues de {self.request_timeout} s con DeepSeek")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Error de conexion con DeepSeek - verifica tu internet")
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                status_code = e.response.status_code
                raise RuntimeError(f"Error HTTP {status_code} con DeepSeek: {str(e)}")
            else:
                raise RuntimeError(f"Error de conexión con DeepSeek: {str(e)}")
        except KeyError as e:
            raise RuntimeError(f"Respuesta inesperada de DeepSeek: {e}")
        except Exception as e:
            raise RuntimeError(f"Error inesperado con DeepSeek: {e}")

    def estimate_cost(self, input_tokens: int, output_tokens: int = 0) -> float:
        """Estima el costo basado en los precios de DeepSeek."""
        if self.model in self.SUPPORTED_MODELS:
            prices = self.SUPPORTED_MODELS[self.model]
            input_cost = (input_tokens / 1000) * prices["input"]
            output_cost = (output_tokens / 1000) * prices["output"]
            return input_cost + output_cost
        else:
            # Fallback para modelos no listados
            return (input_tokens / 1_000_000) * 0.10

    def _validate_connection(self):
        """Método de compatibilidad (ya no se usa automáticamente)."""
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Retorna información sobre el modelo actual."""
        model_info = self.SUPPORTED_MODELS.get(self.model, {})
        return {
            "provider": "DeepSeek",
            "model": self.model,
            "name": model_info.get("name", "Desconocido"),
            "input_cost": model_info.get("input", 0.00014),
            "output_cost": model_info.get("output", 0.00028),
        }