import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

# Mapping model_name -> nom du fichier .npy
EMBEDDING_FILES = {
    "bow":      "embeddings_bow.npy",
    "word2vec": "embeddings_w2v.npy",
    "bert":     "embeddings_bert.npy",
}


EXTRA_FILES = {
    "bow":      "vectorizer.pkl",
    "word2vec": None,
    "bert":     None,
}

class MovieRecommender:
    """
    Moteur de recommandation basé sur la similarité cosinus entre embeddings.
    Les embeddings sont toujours chargés depuis les fichiers pré-calculés.
    """
    def __init__(self, embedder, df: pd.DataFrame, model_name: str):
        if model_name not in EMBEDDING_FILES:
            raise ValueError(f"model_name doit être parmi {list(EMBEDDING_FILES.keys())}")
        self.embedder = embedder
        self.df = df
        self.embeddings = None
        self.model_name = model_name

    def load (self, base_path: str = "saved_models"):
        """Charge les embeddings pré-calculés depuis le fichier correspondant."""
        filename = EMBEDDING_FILES[self.model_name]
        file_path = os.path.join(base_path, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Embeddings introuvables : {file_path}\n"
                f"Lance precompute_embeddings.py ou télécharge les fichiers depuis Drive."
            )

        self.embeddings = np.load(file_path)
        # Synchronise aussi l'embedder (nécessaire pour transform() sur la query)
        self.embedder.embeddings = self.embeddings
        print(f"[{self.model_name}] Embeddings chargés : {self.embeddings.shape}")
       
        # 2. Charge le fichier supplémentaire si nécessaire (vectorizer pour BoW, modèle pour Word2Vec)
        extra = EXTRA_FILES[self.model_name]
        if extra is not None:
            extra_path = os.path.join(base_path, extra)
            if not os.path.exists(extra_path):
                raise FileNotFoundError(f"Fichier supplémentaire introuvable : {extra_path}")
            self.embedder.load_extra(extra_path)
            print(f"[{self.model_name}] Fichier supplémentaire chargé : {extra_path}")



    def recommend(self, query: str, top_k: int = 5) -> pd.DataFrame:
        if self.embeddings is None:
            raise RuntimeError("Embeddings non chargés — appelle load_index() d'abord.")

        # top_k cappé à la taille du dataset, cohérent avec l'API
        top_k = min(top_k, len(self.df))

        query_embedding = self.embedder.transform([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        return results[['movie_poster_path', 'movie_category', 'movie_plot', 'similarity_score']]