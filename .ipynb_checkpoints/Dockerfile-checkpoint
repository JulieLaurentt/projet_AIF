# 1. Utiliser Python 3.10 (stable pour PyTorch et Gradio)
FROM python:3.10-slim

# 2. Définir le dossier de travail dans le conteneur
WORKDIR /app

# 3. Installer les dépendances système nécessaires (OpenCV, Pillow, etc.)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copier et installer les bibliothèques Python
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
# On force l'installation de Gradio et Requests au cas où
RUN pip install --no-cache-dir gradio requests 

# 5. Copier tout ton code (le modèle .pth doit être dans un dossier 'weights/')
COPY . .

# 6. Exposer les ports pour l'extérieur
# 5075 = Flask (API) | 7860 = Gradio (Interface)
EXPOSE 5075 7860

# 7. La commande de lancement
# On utilise 'sh -c' pour lancer deux processus en même temps :
# 'python movieposter_api.py &' lance l'API en tâche de fond.
# 'python app_gradio.py' lance l'interface au premier plan.
CMD sh -c "python movieposter_api.py --model_path weights/movieposter_net.pth & python app_gradio.py"