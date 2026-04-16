import numpy as np
from sentence_transformers import SentenceTransformer
import os

class BERTEmbedder:
    """
    Sentence-BERT : encode chaque synopsis en un vecteur dense de 384 dims.
    C'est l'approche transformer du TP, adaptée à la recommandation.
    Modèle utilisé : all-MiniLM-L6-v2 (rapide + performant)
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None

    def fit_transform(self, plots: list[str]) -> np.ndarray:
        """Encode tous les synopsis. batch_size=32 pour éviter les OOM."""
        self.embeddings = self.model.encode(
            plots,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True  # important pour la similarité cosinus !
        )
        return self.embeddings

    def transform(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)

    def save(self, path: str = "saved_models/bert"):
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/embeddings.npy", self.embeddings)

    def load(self, path: str = "saved_models/bert"):
        self.embeddings = np.load(f"{path}/embeddings.npy")