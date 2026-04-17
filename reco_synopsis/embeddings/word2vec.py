import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk, os, pickle


from  clean_embedd import TextPreprocessor

class Word2VecEmbedder:
    def __init__(self, vector_size=100, window=5, min_count=2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
        self.embeddings = None
        self.preprocessor = TextPreprocessor(use_lemmatization=True)  # <-- remplace l'ancienne tokenisation

    def fit_transform(self, plots: list[str]) -> np.ndarray:
        tokenized = self.preprocessor.preprocess_batch_tokens(plots)  # <-- plus propre
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
        tokenized = self.preprocessor.preprocess_batch_tokens(texts)  # <-- ajout
        return np.array([self._mean_vector(t) for t in tokenized])
    
    def save(self, path: str = "saved_models/w2v"):
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/embeddings.npy", self.embeddings)

    def load(self, path: str = "saved_models/w2v"):
        self.embeddings = np.load(f"{path}/embeddings.npy")
