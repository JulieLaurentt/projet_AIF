import gradio as gr
import requests
import io
import os
from PIL import Image
import time

# --- Fonctions utilitaires ---

def call_api_with_retry(url, **kwargs):
    for i in range(10):
        try:
            response = requests.post(url, **kwargs)
            return response
        except requests.exceptions.ConnectionError:
            time.sleep(3)
    return None

# =========================================================
# ONGLET 1 — Prédiction de genre par poster
# =========================================================

def predict_movie_genre(image):
    api_url = os.getenv("API_CLASSIFICATION_URL", "http://localhost:5075/predict")

    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_data = img_byte_arr.getvalue()

    response = call_api_with_retry(api_url, data=img_data)

    if response is None:
        return "API de classification non disponible après 10 tentatives."

    if response.status_code == 200:
        prediction = response.json().get('label', 'Genre inconnu')
        return f"Genre prédit : {prediction}"
    else:
        return f"Erreur API : Code {response.status_code}"

# =========================================================
# ONGLET 2 — Recommandation par synopsis
# =========================================================

def recommend_movies(query, method, top_k):
    # Correction de l'URL (suppression du double http://)
    api_url = os.getenv("API_RECOMMENDATION_URL", "http://localhost:5076/recommend")

    if not query.strip():
        return "Veuillez entrer une description de film."

    response = call_api_with_retry(
        api_url,
        json={
            "query": query.strip(),
            "method": method,
            "top_k": int(top_k)
        }
    )

    if response is None:
        return "API de recommandation non disponible après 10 tentatives."

    if response.status_code != 200:
        return f"Erreur API : Code {response.status_code}"

    results = response.json().get("results", [])

    if not results:
        return "Aucun résultat trouvé."

    # Construction de la galerie de résultats en HTML
    html_output = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;'>"
    for r in results:
        poster_path = r.get("movie_poster_path", "")
        # Gradio utilise le préfixe 'file/' pour accéder aux fichiers locaux
        img_src = f"file/{poster_path}" if poster_path else ""
        category = r.get("movie_category", "")
        score = r.get("similarity_score", 0)
        plot = r.get("movie_plot", "")
        
        html_output += f"""
        <div style='border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <img src='{img_src}' style='width: 100%; height: auto; border-radius: 4px;' onerror="this.src='https://via.placeholder.com/200x300?text=Pas+d%27affiche'">
            <div style='margin-top: 10px;'>
                <div style='font-weight: bold; color: #333;'>{category}</div>
                <div style='color: #007bff; font-size: 0.9em; margin: 5px 0;'>Score : {score:.4f}</div>
                <p style='font-size: 0.8em; color: #666; text-align: justify; line-height: 1.3;'>{plot[:150]}...</p>
            </div>
        </div>
        """
    html_output += "</div>"
    return html_output

# =========================================================
# Interface Gradio — 2 onglets
# =========================================================

with gr.Blocks(title="Analyseur de Films") as demo:

    gr.Markdown("# Analyseur de Films")

    with gr.Tabs():

        # --- Onglet 1 ---
        with gr.Tab("Prédiction de genre"):
            gr.Markdown("### Prédire le genre d'un film à partir de son poster")
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Poster")
                    predict_btn = gr.Button("Analyser", variant="primary")
                with gr.Column():
                    genre_output = gr.Text(label="Résultat")

            predict_btn.click(
                fn=predict_movie_genre,
                inputs=image_input,
                outputs=genre_output
            )

        # --- Onglet 2 ---
        with gr.Tab("Recommandation par synopsis"):
            gr.Markdown("### Trouver des films similaires")
            
            # Formulaire de saisie
            with gr.Column():
                query_input = gr.Textbox(
                    lines=4,
                    placeholder="Ex: A hero fights to save the world...",
                    label="Description / Synopsis"
                )
                with gr.Row():
                    method_input = gr.Radio(
                        choices=["bow", "word2vec", "bert"],
                        value="bert",
                        label="Méthode d'embedding"
                    )
                    topk_input = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Nombre de recommandations"
                    )
                reco_btn = gr.Button("Recommander", variant="primary")

            # Zone de résultats affichée en dessous
            reco_output = gr.HTML()

            reco_btn.click(
                fn=recommend_movies,
                inputs=[query_input, method_input, topk_input],
                outputs=reco_output
            )

if __name__ == "__main__":
    # L'argument allowed_paths est nécessaire pour servir les images locales situées dans le répertoire courant ou ses sous-dossiers
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["."])