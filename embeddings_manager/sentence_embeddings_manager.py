from sentence_transformers import SentenceTransformer

class SentenceEmbeddingsManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the SentenceEmbeddingsManager with a specified model name.
        
        Args:
            model_name (str): The name of the sentence transformer model to use.
        """
        self.model = SentenceTransformer(model_name)
    
    def sentence_to_embedding(self, sentence: str) -> list[float]:

        return self.model.encode(sentence)
    
    