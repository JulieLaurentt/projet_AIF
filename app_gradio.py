import gradio as gr
import requests
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import io
import os
import time
import base64
from PIL import Image
from torchvision import models 

# =========================================================
# CONFIG URLs
# =========================================================
API_CLASSIFICATION_URL = os.getenv("API_CLASSIFICATION_URL", "http://localhost:5075/predict")
API_RECOMMENDATION_URL = os.getenv("API_RECOMMENDATION_URL", "http://localhost:5076/recommend")
ANNOY_URL              = os.getenv("ANNOY_URL",              "http://annoy:5000/recommend")
ANNOY_CLIP_URL         = os.getenv("ANNOY_CLIP_URL",         "http://annoy_clip:5077")


# =========================================================
# UTILITAIRES
# =========================================================

def call_api_with_retry(url, **kwargs):
    for i in range(10):
        try:
            response = requests.post(url, **kwargs)
            return response
        except requests.exceptions.ConnectionError:
            time.sleep(3)
    return None

def normalize_vector(vector):
    vector = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    return (vector / norm).tolist() if norm > 0 else vector

def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def render_movie_cards(results, section_title):
    """Génère du HTML pour une liste de films {title, plot, image_b64}"""
    if not results:
        return f"<p style='color:#888;'>Aucun résultat pour : {section_title}</p>"

    html = f"<h3 style='color:#e67e22; margin:16px 0 8px;'>{section_title}</h3>"
    html += "<div style='display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:16px;'>"
    for i, r in enumerate(results):
        title   = r.get("title", f"Film {i+1}")
        img_b64 = r.get("image_b64", "")
        if img_b64:
            img_tag = f"<img src='data:image/jpeg;base64,{img_b64}' style='width:100%; height:240px; object-fit:cover; border-radius:4px; margin-bottom:8px;'>"
        else:
            img_tag = "<div style='width:100%; height:240px; background:#eee; border-radius:4px; margin-bottom:8px; display:flex; align-items:center; justify-content:center; color:#aaa;'>Pas d'affiche</div>"
        html += f"""
        <div style='border:1px solid #ddd; border-radius:8px; padding:10px;
                    background:#fff; box-shadow:0 2px 4px rgba(0,0,0,0.08);'>
            {img_tag}
            <div style='font-weight:bold; color:#2c3e50; font-size:0.85em;'>{title}</div>
        </div>"""
    html += "</div>"
    return html

# =========================================================
# ONGLET 1 — Pred
# =========================================================

def predict_movie_genre(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_data = img_byte_arr.getvalue()

    response = call_api_with_retry(API_CLASSIFICATION_URL, data=img_data)

    if response is None:
        return " API de classification non disponible."
    if response.status_code == 200:
        prediction = response.json().get('label', 'Genre inconnu')
        return f"🎬 Genre prédit : {prediction}"
    return f" Erreur API : Code {response.status_code}"

# =========================================================
# ONGLET 2 — Recommandation par image/Annoy
# =========================================================

# =========================================================
# FEATURE EXTRACTOR pour recup l'image correctement (MobileNetV2)
# =========================================================
device = "cpu"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # On utilise le même modèle que le TP sinon prob de dim
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        # On garde exactement la même structure : features -> avgpool -> flatten
        self.model = nn.Sequential(
            mobilenet.features, 
            mobilenet.avgpool, 
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)

extractor = FeatureExtractor().to(device)
extractor.eval()

def extract_vector(image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = extractor(tensor).squeeze().cpu().numpy()
    return normalize_vector(vec)

def get_recommendations(image):
    if image is None:
        return "<p>Veuillez uploader une image.</p>"

    # extraction embedding 576
    vector = extract_vector(image)

    # appel API Annoy
    response = call_api_with_retry(
        ANNOY_URL,
        json={"vector": vector}
    )

    if response is None:
        return "<p>API Annoy indisponible.</p>"
    
    if response.status_code != 200:
        return f"<p>Erreur API : {response.status_code}</p>"


    results = response.json().get("results", [])

    films = []

    for movie in results[:5]: 
        try:
            img_bytes = requests.get(movie["image_url"]).content
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            
            films.append({
                "image_b64": img_b64
            })
        except Exception as e:
            print(f"Erreur image: {e}")

    return render_movie_cards(films, "🎬 Films similaires")

# =========================================================
# ONGLET 3 — Recommandation par synopsis
# =========================================================
def recommend_movies(query, method, top_k):
    if not query.strip():
        return "Veuillez entrer une description de film en anglais"

    response = call_api_with_retry(
        API_RECOMMENDATION_URL,
        json={"query": query.strip(), "method": method, "top_k": int(top_k)}
    )

    if response is None:
        return " API de recommandation non disponible."
    if response.status_code != 200:
        return f" Erreur API : Code {response.status_code}"

    results = response.json().get("results", [])
    if not results:
        return "Aucun résultat trouvé."

    html_output = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; margin-top: 20px;'>"

    for i, r in enumerate(results):
        poster_path = r.get("movie_poster_path", "")
        img_src = f"http://localhost:5076/poster/{poster_path}" if poster_path else "https://via.placeholder.com/200x300?text=Pas+d%27affiche"
        category = r.get("movie_category", "")
        score = r.get("similarity_score", 0)
        plot = r.get("movie_plot", "")
        plot_short = plot[:120] + "..." if len(plot) > 120 else plot

        html_output += f"""
        <div style='border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <img src='{img_src}' style='width: 100%; height: 280px; object-fit: cover; border-radius: 4px;'
                onerror="this.src='https://via.placeholder.com/200x300?text=Pas+d%27affiche'">
            <div style='margin-top: 10px;'>
                <div style='font-weight: bold; color: #333;'>{category}</div>
                <div style='color: #007bff; font-size: 0.9em; margin: 5px 0;'>Score : {score:.4f}</div>
                <p id='short_{i}' style='font-size: 0.8em; color: #666; line-height: 1.3; margin: 0;'>
                    {plot_short}
                </p>
                <p id='full_{i}' style='font-size: 0.8em; color: #666; line-height: 1.3; margin: 0; display: none;'>
                    {plot}
                </p>
                <button id='btn_{i}'
                    onclick="
                        var s = document.getElementById('short_{i}');
                        var f = document.getElementById('full_{i}');
                        var b = document.getElementById('btn_{i}');
                        if (f.style.display === 'none') {{
                            s.style.display = 'none';
                            f.style.display = 'block';
                            b.textContent = '−';
                        }} else {{
                            s.style.display = 'block';
                            f.style.display = 'none';
                            b.textContent = '+';
                        }}
                    "
                    style='margin-top: 6px; background: none; border: 1px solid #007bff; color: #007bff;
                        border-radius: 50%; width: 22px; height: 22px; cursor: pointer; font-size: 14px;
                        display: flex; align-items: center; justify-content: center; padding: 0;'>
                    +
                </button>
            </div>
        </div>"""

    html_output += "</div>"
    return html_output

# =========================================================
# ONGLET 4 — Natural Language Movie Discovery (CLIP)
# =========================================================

def clip_recommend(text_query, poster_image, top_k):
    """
    Retourne 3 sections HTML :
      1. Texte → texte  (synopsis similaires)
      2. Image → image  (posters similaires)
      3. Texte → image  (posters correspondant au texte)
    """
    top_k = int(top_k)
    sections_html = ""

    # ── 1. Texte → texte ──────────────────────────────────────────────────
    if text_query and text_query.strip():
        resp = call_api_with_retry(
            f"{ANNOY_CLIP_URL}/recommend/text_from_text",
            json={"query": text_query.strip(), "top_k": top_k}
        )
        if resp is None:
            sections_html += "<p>❌ API CLIP non disponible (text→text).</p>"
        elif resp.status_code == 200:
            results = resp.json().get("results", [])
            sections_html += render_movie_cards(results, "📝 Recommandation texte → texte (synopsis similaires)")
        else:
            sections_html += f"<p>❌ Erreur text→text : {resp.status_code}</p>"
    else:
        sections_html += "<p style='color:#aaa;'>📝 <em>Entrez un synopsis pour la recommandation texte→texte.</em></p>"

    # ── 2. Image → image ──────────────────────────────────────────────────
    if poster_image is not None:
        b64 = image_to_base64(poster_image)
        resp = call_api_with_retry(
            f"{ANNOY_CLIP_URL}/recommend/image_from_image",
            json={"image_b64": b64, "top_k": top_k}
        )
        if resp is None:
            sections_html += "<p>❌ API CLIP non disponible (image→image).</p>"
        elif resp.status_code == 200:
            results = resp.json().get("results", [])
            sections_html += render_movie_cards(results, "🖼️ Recommandation image → image (posters similaires)")
        else:
            sections_html += f"<p>❌ Erreur image→image : {resp.status_code}</p>"
    else:
        sections_html += "<p style='color:#aaa;'>🖼️ <em>Uploadez un poster pour la recommandation image→image.</em></p>"

    # ── 3. Texte → image ──────────────────────────────────────────────────
    if text_query and text_query.strip():
        resp = call_api_with_retry(
            f"{ANNOY_CLIP_URL}/recommend/image_from_text",
            json={"query": text_query.strip(), "top_k": top_k}
        )
        if resp is None:
            sections_html += "<p>❌ API CLIP non disponible (text→image).</p>"
        elif resp.status_code == 200:
            results = resp.json().get("results", [])
            sections_html += render_movie_cards(results, "🔍 Recommandation texte → image (posters correspondant au texte)")
        else:
            sections_html += f"<p>❌ Erreur text→image : {resp.status_code}</p>"
    else:
        sections_html += "<p style='color:#aaa;'>🔍 <em>Entrez un synopsis pour la recommandation texte→image.</em></p>"

    return sections_html or "<p>Aucun résultat.</p>"

# =========================================================
# INTERFACE GRADIO — 4 onglets
# =========================================================

with gr.Blocks(title="AI Movie Analysis") as demo:
    gr.Markdown("#  Analyseur de Films")

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
            predict_btn.click(fn=predict_movie_genre, inputs=image_input, outputs=genre_output)

        # --- Onglet 2 ---
        with gr.Tab("Recommandation par image"):
            gr.Markdown("### Trouver des films similaires à partir d'un poster")
            with gr.Row():
                with gr.Column():
                    image_input_reco = gr.Image(type="pil", label="Poster")
                    annoy_btn = gr.Button("Trouver des films similaires", variant="primary")
                with gr.Column():
                    annoy_output = gr.HTML(label="Films similaires")
            annoy_btn.click(fn=get_recommendations, inputs=image_input_reco, outputs=annoy_output)

        # --- Onglet 3 ---
        with gr.Tab("Recommandation par synopsis"):
            gr.Markdown("### Trouver des films similaires à partir d'une description en anglais")
            with gr.Column():
                query_input = gr.Textbox(
                    lines=4,
                    placeholder="Ex: A hero fights to save the world...",
                    label="Description / Synopsis en anglais"
                )
                with gr.Row():
                    method_input = gr.Radio(
                        choices=["bow", "word2vec", "bert"],
                        value="bert",
                        label="Méthode d'embedding"
                    )
                    topk_input = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Nombre de recommandations")
                reco_btn = gr.Button("Recommander", variant="primary")
            reco_output = gr.HTML()
            reco_btn.click(fn=recommend_movies, inputs=[query_input, method_input, topk_input], outputs=reco_output)

        # --- Onglet 4 :  ---
        with gr.Tab("🎬 Découverte CLIP"):
            gr.Markdown("""
            ### Natural Language Movie Discovery (CLIP)
            Utilisez le langage naturel ou un poster pour découvrir des films similaires.
            CLIP encode texte et images dans le même espace vectoriel.
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    clip_text = gr.Textbox(
                        lines=4,
                        placeholder="Ex: a young wizard discovers his powers and goes to a magical school...",
                        label="📝 Description / Synopsis en anglais"
                    )
                    clip_image = gr.Image(
                        type="pil",
                        label="🖼️ Poster (optionnel, pour reco image→image)"
                    )
                    clip_topk = gr.Slider(minimum=1, maximum=10, value=5, step=1,
                                          label="Nombre de recommandations par section")
                    clip_btn = gr.Button("🔍 Découvrir", variant="primary")

            gr.Markdown("""
            **3 types de recommandations retournées :**
            - 📝 **Texte → Texte** : films dont le synopsis est sémantiquement proche
            - 🖼️ **Image → Image** : films dont le poster est visuellement similaire
            - 🔍 **Texte → Image** : films dont le poster correspond à votre description
            """)

            clip_output = gr.HTML()
            clip_btn.click(
                fn=clip_recommend,
                inputs=[clip_text, clip_image, clip_topk],
                outputs=clip_output
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["."])