import gradio as gr
import requests
import io
import os # N'oublie pas d'importer os
from PIL import Image

def predict_movie_genre(image):
    # Utilise la variable d'environnement ou 'api' par défaut (qui sera le nom du service Docker)
    api_base_url = os.getenv("API_URL", "http://api:5075/predict")
    
    # 2. Conversion de l'image Gradio (PIL) en bytes pour l'API
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_data = img_byte_arr.getvalue()

    try:
        # 3. Envoi de la requête
        response = requests.post(api_base_url, data=img_data)
        
        if response.status_code == 200:
            prediction = response.json().get('label', 'Genre inconnu')
            return f"🎬 Genre prédit : {prediction}"
        else:
            return f"⚠️ Erreur API : Code {response.status_code}"
            
    except Exception as e:
        return f"Impossible de contacter l'API. ({e})"

# 4. Création de l'interface visuelle
demo = gr.Interface(
    fn=predict_movie_genre,
    inputs=gr.Image(type="pil", label="Déposez un poster ici"),
    outputs=gr.Text(label="Résultat de l'analyse"),
    title="Analyseur de Posters de Films",
    description="Cette interface utilise une API Flask et un modèle Deep Learning pour prédire le genre d'un film."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)