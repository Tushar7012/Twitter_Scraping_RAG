from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """Generates vector embeddings for text."""
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded.")

    def generate_embedding(self, text: str):
        """Generates a vector embedding for the given text."""
        return self.model.encode(text)