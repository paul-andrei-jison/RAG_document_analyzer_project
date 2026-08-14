from typing import Protocol, List, runtime_checkable
import ollama

@runtime_checkable
class AIProvider(Protocol):
    """
    Interface definition for any AI backend.
    Any provider (Ollama, Claude, OpenAI, Bedrock) must implement these methods.
    """
    def generate(self, prompt: str) -> str:
        """Takes a text prompt and returns the LLM-generated string."""
        ...

    def embed(self, text: str) -> List[float]:
        """Takes text and returns a vector embedding as a list of floats."""
        ...


class OllamaProvider:
    """
    Concrete implementation of AIProvider using local Ollama models.
    """
    def __init__(
        self, 
        llm_model: str = "llama3.2", 
        embed_model: str = "nomic-embed-text"
    ):
        self.llm_model = llm_model
        self.embed_model = embed_model
        # Ollama's Client will connect to the default http://localhost:11434
        self.client = ollama.Client()

    def generate(self, prompt: str) -> str:
        """Generates text from the local LLM model."""
        response = self.client.generate(
            model=self.llm_model, 
            prompt=prompt
        )
        return response["response"]

    def embed(self, text: str) -> List[float]:
        """Generates embeddings using the local embedding model."""
        response = self.client.embed(
            model=self.embed_model, 
            input=text
        )
        # nomic-embed-text returns a list of embeddings; grab the first one
        return response["embeddings"][0]


# --- Provider Factory ---
def get_ai_provider(provider_type: str = "ollama") -> AIProvider:
    """
    Factory function to return the configured AI provider.
    Later you can add 'anthropic', 'bedrock', etc. here based on config.
    """
    if provider_type.lower() == "ollama":
        return OllamaProvider()
    else:
        raise ValueError(f"Unsupported AI Provider type: {provider_type}")