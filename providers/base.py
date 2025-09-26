from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseProvider(ABC):
    name: str

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        pass    