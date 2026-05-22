import pandas as pd
from embeddings.bow import BoWEmbedder

df = pd.read_csv("data/movie_plots.csv")
plots = df['movie_plot'].fillna('').tolist()

bow = BoWEmbedder()
bow.fit_transform(plots)

# Sauvegarde uniquement le vectorizer 
import pickle
with open("saved_models/vectorizer.pkl", 'wb') as f:
    pickle.dump(bow.vectorizer, f)

print("vectorizer.pkl régénéré !")