import ollama

def verify_setup():
    print("Testing connection to Ollama...")
    
    # 1. Test Text Generation (LLM)
    print("\n--- Testing llama3.2 Generation ---")
    try:
        gen_response = ollama.generate(
            model='llama3.2', 
            prompt='Explain RAG (Retrieval-Augmented Generation) in one short sentence.'
        )
        print("Success! Response:")
        print(f"> {gen_response['response'].strip()}")
    except Exception as e:
        print(f"Error calling llama3.2: {e}")
        return

    # 2. Test Embeddings
    print("\n--- Testing nomic-embed-text Embeddings ---")
    try:
        # Note: Depending on your Ollama python client version, it might be 
        # ollama.embed() or ollama.embeddings(). We will use embed().
        embed_response = ollama.embed(
            model='nomic-embed-text', 
            input='This is a test document.'
        )
        vector = embed_response.get('embeddings', [[]])[0]
        
        print("Success! Embedding generated.")
        print(f"Vector dimension size: {len(vector)} (Expected 768 for nomic-embed-text)")
    except Exception as e:
        print(f"Error calling nomic-embed-text: {e}")
        return

    print("\n✅ All systems go! Environment and models are ready.")

if __name__ == "__main__":
    verify_setup()