"""
annoy_clip.py — Service Flask CLIP avec affichage des posters
"""

import json, os, io, base64, torch, numpy as np
from flask import Flask, request, jsonify
from annoy import AnnoyIndex
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

app = Flask(__name__)

MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_DIM   = 512

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model     = CLIPModel.from_pretrained(MODEL_NAME).to(device)
model.eval()
print(f"✅ CLIP chargé sur {device}")

index_posters = AnnoyIndex(CLIP_DIM, 'angular')
index_plots   = AnnoyIndex(CLIP_DIM, 'angular')
index_posters.load("weights/clip_posters.ann")
index_plots.load("weights/clip_plots.ann")

with open("weights/clip_metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"✅ Index chargés — {len(metadata)} films")

def encode_text(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True,
                       truncation=True, max_length=77).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy().astype(np.float32).tolist()

def encode_image_b64(b64_str):
    img_bytes = base64.b64decode(b64_str)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy().astype(np.float32).tolist()

def get_results(indices):
    results = []
    for i in indices:
        if i < len(metadata):
            m = metadata[i]
            results.append({
                "title":     m["title"],
                "plot":      m.get("plot", ""),
                "image_b64": m.get("image_b64", ""),
            })
    return results

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "movies": len(metadata)})

@app.route('/recommend/text_from_text', methods=['POST'])
def text_from_text():
    data  = request.get_json()
    query = data.get('query', '')
    top_k = int(data.get('top_k', 5))
    if not query.strip():
        return jsonify({"error": "query vide"}), 400
    vector  = encode_text(query)
    indices = index_plots.get_nns_by_vector(vector, top_k + 1)[1:]
    return jsonify({"results": get_results(indices), "type": "text→text"})

@app.route('/recommend/image_from_image', methods=['POST'])
def image_from_image():
    data  = request.get_json()
    top_k = int(data.get('top_k', 5))
    if 'vector' in data:
        vector = data['vector']
    elif 'image_b64' in data:
        vector = encode_image_b64(data['image_b64'])
    else:
        return jsonify({"error": "Fournir 'vector' ou 'image_b64'"}), 400
    indices = index_posters.get_nns_by_vector(vector, top_k + 1)[1:]
    return jsonify({"results": get_results(indices), "type": "image→image"})

@app.route('/recommend/image_from_text', methods=['POST'])
def image_from_text():
    data  = request.get_json()
    query = data.get('query', '')
    top_k = int(data.get('top_k', 5))
    if not query.strip():
        return jsonify({"error": "query vide"}), 400
    vector  = encode_text(query)
    indices = index_posters.get_nns_by_vector(vector, top_k + 1)[1:]
    return jsonify({"results": get_results(indices), "type": "text→image"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5077, debug=False)