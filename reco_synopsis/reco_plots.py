import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

class MovieRecommender:
    """
    Moteur de recommandation basé sur la similarité cosinus entre embeddings.
    Fonctionne avec n'importe lequel des embedders (BoW, Word2Vec, BERT).
    """
    def __init__(self, embedder, df: pd.DataFrame, model_name: str):
        self.embedder = embedder
        self.df = df          
        self.embeddings = None
        self.model_name = model_name

    def load_or_build_index(self, base_path: str = "saved_models"):
        path = f"{base_path}/{self.model_name}"
        file_path = f"{path}/embeddings.npy"

        if os.path.exists(file_path):
            self.embedder.load(path)
            self.embeddings = self.embedder.embeddings
            print(f"Index chargé : {self.embeddings.shape}")
        else:
            plots = self.df['movie_plot'].fillna('').tolist()
            self.embeddings = self.embedder.fit_transform(plots)
            self.embedder.save(path)
            print(f"Index construit et sauvegardé : {self.embeddings.shape}")

    def recommend(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_embedding = self.embedder.transform([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        return results