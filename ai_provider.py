import ollama
from typing import Protocol, List, runtime_checkable


@runtime_checkable
class AIProvider(Protocol):
    """Interface any AI backend must implement."""
    def generate(self, prompt: str) -> str: ...
    def embed(self, text: str) -> List[float]: ...


class OllamaProvider:
    def __init__(self, llm_model: str = "llama3.2", embed_model: str = "nomic-embed-text"):
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.client = ollama.Client()

    def generate(self, prompt: str) -> str:
        response = self.client.generate(model=self.llm_model, prompt=prompt)
        return response["response"]

    def embed(self, text: str) -> List[float]:
        response = self.client.embed(model=self.embed_model, input=text)
        return response["embeddings"][0]


def get_ai_provider(model_id: str = "llama3.2") -> AIProvider:
    return OllamaProvider(llm_model=model_id)
