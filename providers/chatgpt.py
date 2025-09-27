from openai import OpenAI
from .base import BaseProvider
import os

class ChatGPTProvider(BaseProvider):
    def __init__(self, api_key: str = None, base_url: str | None = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), api_base=base_url)
        self.name = "chatgpt"

    def chat(self, messages, model="gpt-4o-mini", temperature=0.2, max_tokens=1024, **kwargs):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False
        )
        text = response.choices[0].message.content
        return {"text": text, "raw": response}