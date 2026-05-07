
import pandas as pd
from flask import Flask, jsonify, request

from reco_plots import MovieRecommender

from embeddings.bow import BoWEmbedder
from embeddings.word2vec import Word2VecEmbedder
from embeddings.bert import BERTEmbedder


app = Flask(__name__)

# --- Chargement du CSV ---
df = pd.read_csv("data/movie_plots.csv")
plots = df['movie_plot'].fillna('').tolist()

# --- Instanciation des embedders ---
bow  = BoWEmbedder()
w2v  = Word2VecEmbedder()
bert = BERTEmbedder()

# --- Chargement des embeddings pré-calculés (pas de recalcul au démarrage) ---
rec_bow  = MovieRecommender(bow,  df, model_name="bow")
rec_bow.load ("saved_models")

rec_w2v  = MovieRecommender(w2v,  df, model_name="word2vec")
rec_w2v.load ("saved_models")

rec_bert = MovieRecommender(bert, df, model_name="bert")
rec_bert.load ("saved_models")

# Map méthode -> recommender, comme CLASSES dans ton API existante
RECOMMENDERS = {
    "bow":      rec_bow,
    "word2vec": rec_w2v,
    "bert":     rec_bert
}


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Body JSON attendu :
    {
        "query": "A hero fights evil forces",
        "method": "bert",       # bow | word2vec | bert
        "top_k": 5              # optionnel, défaut 5
    }
    """
    data = request.get_json()
    query  = data.get("query", "")
    method = data.get("method", "bert")
    top_k  = data.get("top_k", 5)

    if not query:
        return jsonify({"error": "query is required"}), 400

    if method not in RECOMMENDERS:
        return jsonify({"error": f"method must be one of {list(RECOMMENDERS.keys())}"}), 400

    results = RECOMMENDERS[method].recommend(query, top_k=top_k)

    return jsonify({
        "method": method,
        "query": query,
        "results": [
            {
                "movie_poster_path": row["movie_poster_path"],
                "movie_category":   row["movie_category"],
                "movie_plot":       row["movie_plot"],
                "similarity_score": round(float(row["similarity_score"]), 4)
            }
            for _, row in results.iterrows()
        ]
    })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5076, debug=False)  # port différent du 5075


    