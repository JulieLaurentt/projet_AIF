# fichier pour faire embeddings par bow 

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle, os
from clean_embedd import TextPreprocessor

class BoWEmbedder:
    def __init__(self, max_features=10000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.preprocessor = TextPreprocessor(use_lemmatization=True)
        self.embeddings = None

    def fit_transform(self, plots: list[str]) -> np.ndarray:
        cleaned = self.preprocessor.preprocess_batch(plots)  
        self.embeddings = self.vectorizer.fit_transform(cleaned).toarray()
        return self.embeddings

    def transform(self, texts: list[str]) -> np.ndarray:
        cleaned = self.preprocessor.preprocess_batch(texts)   
        return self.vectorizer.transform(cleaned).toarray()
    
    def save(self, path: str = "saved_models/bow"):
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/embeddings.npy", self.embeddings)

    def load(self, path: str = "saved_models/bow"):
        self.embeddings = np.load(f"{path}/embeddings.npy")
