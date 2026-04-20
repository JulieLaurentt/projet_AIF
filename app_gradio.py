import gradio as gr
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import io
import os
import time
from PIL import Image


ANNOY_URL = os.getenv("ANNOY_URL", "http://annoy:5000/recommend")

# Modèle pour extraire le vecteur
model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def call_api(image):
    api_base_url = os.getenv("API_URL", "http://api:5075/predict")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_data = img_byte_arr.getvalue()
    
    for i in range(10):
        try:
            response = requests.post(api_base_url, data=img_data)
            return response.json()
        except requests.exceptions.ConnectionError:
            time.sleep(3)
    return None

def normalize_vector(vector):
    vector = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    return (vector / norm).tolist() if norm > 0 else vector

def call_annoy(vector):
    for i in range(10):
        try:
            vector = normalize_vector(vector)
            response = requests.post(ANNOY_URL, json={"vector": vector})
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.ConnectionError:
            time.sleep(3)
    return None

def predict_movie_genre(image):
    result = call_api(image)
    if result is None:
        return "⚠️ API non disponible"
    prediction = result.get('label', 'Genre inconnu')
    return f"🎬 Genre prédit : {prediction}"

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
        self.features = nn.Sequential(*list(mobilenet.children())[:-1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 1280 dimensions
        return x

def get_recommendations(image):
    # Extraction du vecteur côté Gradio
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        model = FeatureExtractor()
        vector = model(tensor).squeeze().cpu().numpy().tolist()

    annoy_result = call_annoy(vector)
    films = annoy_result.get('recommendations', []) if annoy_result else []
    return "\n".join([f"🎥 {film}" for film in films])

# 4. Création de l'interface visuelle
with gr.Blocks(title="Analyseur de Posters de Films") as demo:
    gr.Markdown("# Analyseur de Posters de Films")
    gr.Markdown("Cette interface utilise une API Flask et un modèle Deep Learning pour prédire le genre d'un film.")
    
    image_input = gr.Image(type="pil", label="Déposez un poster ici")
    genre_output = gr.Text(label="Genre prédit")
    reco_output = gr.Text(label="Films similaires")

    image_input.change(predict_movie_genre, inputs=image_input, outputs=genre_output)
    image_input.change(get_recommendations, inputs=image_input, outputs=reco_output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)