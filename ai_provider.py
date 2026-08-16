import json
from typing import Protocol, List, runtime_checkable
import boto3


@runtime_checkable
class AIProvider(Protocol):
    """Interface any AI backend must implement."""
    def generate(self, prompt: str) -> str: ...
    def embed(self, text: str) -> List[float]: ...


class OllamaProvider:
    def __init__(self, llm_model: str = "llama3.2", embed_model: str = "nomic-embed-text"):
        import ollama
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.client = ollama.Client()

    def generate(self, prompt: str) -> str:
        response = self.client.generate(model=self.llm_model, prompt=prompt)
        return response["response"]

    def embed(self, text: str) -> List[float]:
        response = self.client.embed(model=self.embed_model, input=text)
        return response["embeddings"][0]


class BedrockProvider:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime")

    def generate(self, prompt: str) -> str:
        response = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
        )
        return response["output"]["message"]["content"][0]["text"]

    def embed(self, text: str) -> List[float]:
        body = json.dumps({"inputText": text})
        response = self.client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())["embedding"]


def get_ai_provider(provider_type: str = "ollama", model_id: str = None) -> AIProvider:
    if provider_type.lower() == "ollama":
        return OllamaProvider(llm_model=model_id or "llama3.2")
    elif provider_type.lower() == "bedrock":
        if not model_id:
            raise ValueError("model_id is required for Bedrock provider")
        return BedrockProvider(model_id=model_id)
    else:
        raise ValueError(f"Unsupported AI provider: {provider_type}")
