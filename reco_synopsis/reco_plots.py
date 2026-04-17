import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    """
    Moteur de recommandation basé sur la similarité cosinus entre embeddings.
    Fonctionne avec n'importe lequel des embedders (BoW, Word2Vec, BERT).
    """
    def __init__(self, embedder, df: pd.DataFrame):
        self.embedder = embedder
        self.df = df          # le DataFrame avec movie_plot, movie_category, movie_poster_path
        self.embeddings = None

    def build_index(self):
        """Calcule tous les embeddings du dataset."""
        plots = self.df['movie_plot'].fillna('').tolist()
        self.embeddings = self.embedder.fit_transform(plots)
        print(f"Index construit : {self.embeddings.shape}")

    def recommend(self, query: str, top_k: int = 5) -> pd.DataFrame:
        """
        Prend un synopsis (ou une description libre) en entrée,
        retourne les top_k films les plus proches.
        """
        query_embedding = self.embedder.transform([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = self.df.iloc[top_indices].copy()
        results['similarity_score'] = similarities[top_indices]
        return results[['movie_poster_path', 'movie_category', 'movie_plot', 'similarity_score']]