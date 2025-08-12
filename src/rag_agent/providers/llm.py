"""Language model provider implementations."""

from ..core.exceptions import LLMError


class OpenAIChat:
    """OpenAI Chat completion provider."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def answer(self, prompt: str) -> str:
        """Generate response using OpenAI Chat API."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Responda somente com base no contexto. Se não houver informação suficiente, diga explicitamente que não foi encontrado no documento."},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise LLMError(f"OpenAI chat failed: {e}")


class OllamaChat:
    """Local LLM provider via Ollama (http://localhost:11434)."""
    
    def __init__(self, model: str = "llama3.1:8b", host: str = "http://localhost:11434"):
        import requests
        self.requests = requests
        self.model = model
        self.url = f"{host}/api/chat"

    def answer(self, prompt: str) -> str:
        """Generate response using local Ollama model."""
        try:
            r = self.requests.post(self.url, json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "Responda SOMENTE com base no contexto. Se não houver informação suficiente no contexto, diga: 'Não encontrado nos documentos.'"},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0}
            }, timeout=120)
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            raise LLMError(f"Ollama chat failed: {e}")