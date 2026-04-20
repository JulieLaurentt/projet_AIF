from annoy import AnnoyIndex
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

DIM = 576
annoy_index = AnnoyIndex(DIM, 'angular')
annoy_index.load("weights/rec_imdb.ann")
df = pd.read_csv("weights/movies_metadata.csv")

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    vector = data.get('vector')

    indices = annoy_index.get_nns_by_vector(vector, 6)[1:]  # 5 similaires
    results = df.iloc[indices]['title'].tolist()

    return jsonify({"recommendations": results})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)