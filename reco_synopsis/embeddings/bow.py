# fichier pour faire embeddings par bow 

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle, os

class BoWEmbedder:
    """
    TF-IDF Bag of Words embedder.

    """
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)   # unigrammes + bigrammes
        )
        self.embeddings = None
        self.plots = None

    def fit_transform(self, plots: list[str]) -> np.ndarray:
        """Entraîne le vectorizer sur tous les synopsis et retourne les embeddings."""
        self.plots = plots
        self.embeddings = self.vectorizer.fit_transform(plots).toarray()
        return self.embeddings

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transforme de nouveaux textes (ex: requête utilisateur)."""
        return self.vectorizer.transform(texts).toarray()

    def save(self, path: str = "saved_models/bow"):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        np.save(f"{path}/embeddings.npy", self.embeddings)

    def load(self, path: str = "saved_models/bow"):
        with open(f"{path}/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        self.embeddings = np.load(f"{path}/embeddings.npy")