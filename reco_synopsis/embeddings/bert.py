import numpy as np
import os
from sentence_transformers import SentenceTransformer

class BERTEmbedder:
    """
    Remplacement de l'approche token [CLS] par SentenceTransformers.
    Modèle recommandé pour la recherche asymétrique (requête courte/moyenne -> texte long).
    """
    def __init__(self, model_name: str = 'multi-qa-MiniLM-L6-cos-v1', batch_size: int = 32):
        print(f"Chargement du modèle Sentence-Transformers : {model_name}")
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.embeddings = None

    def fit_transform(self, plots: list[str]) -> np.ndarray:
        # Remplacement des valeurs nulles par des chaînes vides
        plots = [p if isinstance(p, str) else '' for p in plots]
        
        # Encodage avec normalisation (optimise le calcul de la similarité cosinus)
        self.embeddings = self.model.encode(
            plots,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return self.embeddings

    def transform(self, texts: list[str]) -> np.ndarray:
        texts = [t if isinstance(t, str) else '' for t in texts]
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True
        )

    def save(self, path: str = "saved_models/bert"):
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/embeddings.npy", self.embeddings)

    def load(self, path: str = "saved_models/bert"):
        self.embeddings = np.load(f"{path}/embeddings.npy")