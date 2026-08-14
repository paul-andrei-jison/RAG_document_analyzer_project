from ai_provider import get_ai_provider, AIProvider

def test_interface():
    # Instantiate via factory
    provider: AIProvider = get_ai_provider("ollama")
    
    print("Testing Provider Generation...")
    output = provider.generate("Say 'Hello, interface!' and nothing else.")
    print(f"Generated: {output.strip()}")

    print("\nTesting Provider Embedding...")
    vector = provider.embed("Vector abstraction test.")
    print(f"Embedding length: {len(vector)}")
    print(f"Embedding sample (first 3 floats): {vector[:3]}")

if __name__ == "__main__":
    test_interface()