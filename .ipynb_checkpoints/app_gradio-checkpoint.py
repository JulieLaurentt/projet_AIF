import gradio as gr
import requests
import io
from PIL import Image

def predict_movie_genre(image):
    # 1. Configuration (basée sur le test de ta collègue)
    API_URL = "http://127.0.0.1:5075/predict"
    
    # 2. Conversion de l'image Gradio (PIL) en bytes pour l'API
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_data = img_byte_arr.getvalue()

    try:
        # 3. Envoi de la requête (format data brut comme dans son test)
        response = requests.post(API_URL, data=img_data)
        
        if response.status_code == 200:
            prediction = response.json().get('label', 'Genre inconnu')
            return f"🎬 Genre prédit : {prediction}"
        else:
            return f"⚠️ Erreur API : Code {response.status_code}"
            
    except Exception as e:
        return f"Impossible de contacter l'API. Est-elle lancée sur le port 5075 ? ({e})"

# 4. Création de l'interface visuelle
demo = gr.Interface(
    fn=predict_movie_genre,
    inputs=gr.Image(type="pil", label="Déposez un poster ici"),
    outputs=gr.Text(label="Résultat de l'analyse"),
    title="Analyseur de Posters de Films",
    description="Cette interface utilise une API Flask et un modèle Deep Learning pour prédire le genre d'un film."
)

if __name__ == "__main__":
#CRUCIAL pour Docker : server_name="0.0.0.0" permet l'accès extérieur
    demo.launch(server_name="0.0.0.0", server_port=7860)