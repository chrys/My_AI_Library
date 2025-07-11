import ollama

def test_embedding_model(model_name):
    try:
        # Generate an embedding for a simple piece of text
        # The content of the prompt doesn't matter, only the output vector
        response = ollama.embeddings(
            model=model_name,
            prompt='Hello, world!'
        )

        # The embedding is a list of floats
        embedding_vector = response['embedding']

        # The number of dimensions is the length of this list
        dimensions = len(embedding_vector)

        print(f"✅ The embedding dimensions for model '{model_name}' is: {dimensions}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print(f"   Please make sure you have pulled the model with 'ollama pull {model_name}'")


if __name__ == "__main__":
    test_embedding_model("quentinz/bge-base-zh-v1.5:latest")