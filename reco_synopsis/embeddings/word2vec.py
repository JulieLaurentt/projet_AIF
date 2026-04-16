import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk, os, pickle

nltk.download('punkt')
nltk.download('stopwords')

class Word2VecEmbedder:
    """
    Word2Vec : on entraîne un modèle sur les synopsis,
    puis on représente chaque synopsis par la moyenne de ses vecteurs de mots.
    Inspiré de la partie 2 du TP.
    """
    def __init__(self, vector_size=100, window=5, min_count=2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        self.embeddings = None
        self.stop_words = set(stopwords.words('english'))

    def _tokenize(self, text: str) -> list[str]:
        tokens = word_tokenize(text.lower())
        return [t for t in tokens if t.isalpha() and t not in self.stop_words]

    def _mean_vector(self, tokens: list[str]) -> np.ndarray:
        """Moyenne des vecteurs de mots connus du modèle."""
        vectors = [
            self.model.wv[word]
            for word in tokens
            if word in self.model.wv
        ]
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.vector_size)

    def fit_transform(self, plots: list[str]) -> np.ndarray:
        tokenized = [self._tokenize(p) for p in plots]
        self.model = Word2Vec(
            sentences=tokenized,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=10
        )
        self.embeddings = np.array([self._mean_vector(t) for t in tokenized])
        return self.embeddings

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.array([self._mean_vector(self._tokenize(t)) for t in texts])

    def save(self, path: str = "saved_models/word2vec"):
        os.makedirs(path, exist_ok=True)
        self.model.save(f"{path}/word2vec.model")
        np.save(f"{path}/embeddings.npy", self.embeddings)

    def load(self, path: str = "saved_models/word2vec"):
        self.model = Word2Vec.load(f"{path}/word2vec.model")
        self.embeddings = np.load(f"{path}/embeddings.npy")