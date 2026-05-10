from annoy import AnnoyIndex
from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import os

app = Flask(__name__)

# Configuration (dim pareil que dans le TP)
DIM = 576
POSTERS_DIR = "MLP-20M" 
CSV_PATH = "data/mapping_annoy.csv"

# --- Chargement de l'index Annoy ---
annoy_index = AnnoyIndex(DIM, 'angular')
annoy_index.load("weights/rec_imdb.ann")

# --- Chargement des métadonnées (juste les paths) ---
df = pd.read_csv(CSV_PATH, low_memory=False) 

# Route pour servir les images localement depuis le dossier dézippé 
@app.route('/posters/<path:filename>')
def serve_poster(filename):
    return send_from_directory(POSTERS_DIR, filename)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    
    if not data or "vector" not in data:
        return jsonify({"error": "vector missing"}), 400

    vector = data["vector"]
    if len(vector) != DIM:
        return jsonify({"error": f"vector must be {DIM}D"}), 400

    indices = annoy_index.get_nns_by_vector(vector, 6)
    recommendation_indices = indices[1:] 

    results = []
    base_url = request.host_url 

    for idx in recommendation_indices:
        if idx < len(df):
            row = df.iloc[idx]
            poster_path = row.get('path', '')
            filename = poster_path.split('\\')[-1]  # Dernier élément après \
            results.append({
                "image_url": f"{base_url}posters/{filename}"
            })

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)