from abc import ABC, abstractmethod
import os
import requests
from typing import List, Dict


class BaseModelInterface(ABC):
    """Abstract base class for model interfaces."""

    @abstractmethod
    def generate_response(
        self, messages: List[Dict[str, str]], max_length: int = 2048
    ) -> str:
        """Generate a response given a list of messages."""
        pass


class OllamaInterface(BaseModelInterface):
    """Interface for Ollama API."""

    def __init__(self, base_url="http://localhost:11434/api/chat", model="llama3"):
        self.base_url = base_url
        self.model = model

    def generate_response(
        self, messages: List[Dict[str, str]], max_length: int = 2048
    ) -> str:
        try:
            response = requests.post(
                self.base_url,
                json={"model": self.model, "messages": messages, "stream": False},
                timeout=3600,
            )

            if response.status_code == 200:
                data = response.json()
                return data["message"]["content"]
            else:
                raise RuntimeError(f"Error: Status code {response.status_code}")

        except (requests.exceptions.RequestException, KeyError) as e:
            raise RuntimeError(f"Error in generate_response: {str(e)}")


_model = None


def get_model() -> BaseModelInterface:
    """Get or create the model instance for the current process."""
    global _model
    if _model is None:
        _model = OllamaInterface()
    return _model
