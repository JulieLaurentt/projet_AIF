import numpy as np
import gensim.downloader as api
import os
from embeddings.clean_embedd import TextPreprocessor

class Word2VecEmbedder:
    def __init__(self, model_name='glove-wiki-gigaword-100'):
        # Télécharge et charge un modèle pré-entraîné de 100 dimensions (environ 130 Mo)
        # Options alternatives : 'word2vec-google-news-300' (plus lourd)
        print(f"Chargement du modèle pré-entraîné {model_name}...")
        self.model = api.load(model_name)
        self.vector_size = self.model.vector_size
        self.embeddings = None
        self.preprocessor = TextPreprocessor(use_lemmatization=True)

    def _mean_vector(self, tokens: list[str]) -> np.ndarray:
        # Conserver uniquement les tokens présents dans le vocabulaire du modèle
        valid_tokens = [t for t in tokens if t in self.model]
        
        if not valid_tokens:
            return np.zeros(self.vector_size)
            
        # Moyenne simple des vecteurs valides
        return np.mean([self.model[t] for t in valid_tokens], axis=0)

    def fit_transform(self, plots: list[str]) -> np.ndarray:
        tokenized = self.preprocessor.preprocess_batch_tokens(plots)
        self.embeddings = np.array([self._mean_vector(t) for t in tokenized])
        return self.embeddings

    def transform(self, texts: list[str]) -> np.ndarray:
        tokenized = self.preprocessor.preprocess_batch_tokens(texts)
        return np.array([self._mean_vector(t) for t in tokenized])
    
    def save(self, path: str = "saved_models/word2vec"):
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/embeddings.npy", self.embeddings)

    def load(self, path: str = "saved_models/word2vec"):
        self.embeddings = np.load(f"{path}/embeddings.npy")