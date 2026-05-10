---
title: Projet AIF Movie Recommendation System
emoji: 🎬
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# 🎬 Projet AIF - Movie Recommendation System

Un système complet de recommandation de films utilisant plusieurs stratégies: classification de genre par image, recommandation par similarité visuelle, recherche par synopsis et découverte avec CLIP.

---

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Fonctionnalités](#fonctionnalités)
- [Installation & Déploiement](#installation--déploiement)
- [Structure du projet](#structure-du-projet)
- [Configuration](#configuration)
- [Fichier clés](#fichier--clés)

---

## 🎯 Vue d'ensemble

Ce projet implémente un **système de recommandation de films multi-modal** avec une interface Gradio intuitive. Il combine:

- **Deep Learning** (MobileNet v3, CLIP)
- **Vector Databases** (Annoy pour recherche ANN rapide)
- **NLP** (Analyse de synopsis)
- **APIs Flask** pour inférence et recommandations

### Stack Technique

- **Frontend**: Gradio (interface web interactive)
- **Backend**: Flask (APIs de recommandation)
- **ML Models**: PyTorch (MobileNet v3 Small, CLIP)
- **Database**: Annoy (recherche par similarité)
- **Containerization**: Docker & Docker Compose

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    GRADIO (Frontend)                │
│              Interface utilisateur web               │
└─────────────────┬───────────────────────────────────┘
                  │
        ┌─────────┼─────────┬──────────
        │         │         │          │         
   ┌────▼──┐ ┌───▼──┐ ┌───▼──┐ ┌────▼──┐ 
   │ API   │ │Annoy │ │Annoy │ │ API   │ 
   │Class. │ │ Reco │ │CLIP  │ │ Reco │ 
   │(5075) │ │(5000)│ │(5077)│ │(5076)│ 
   └───────┘ └──────┘ └──────┘ └──────┘ 
   Flask    Flask    Flask    Flask    
```

---

## ✨ Fonctionnalités

### 1️⃣ **Prédiction de Genre** 📺
- Upload un poster de film
- Prédiction du genre (Drama, Action, Comedy, etc.)
- Modèle: Classification CNN

### 2️⃣ **Recommandation par Image** 🖼️
- Upload un poster (ta propre affiche ou d'un film existant)
- Extraction des features avec **MobileNet v3 Small** (576D)
- Recherche des **5 films similaires** via **Annoy** (ANN)
- Affichage des posters recommandés

### 3️⃣ **Recommandation par Synopsis** 📖
- Écris ou colle la description d'un film
- Analyse NLP avec **embeddings texte**
- Recherche dans la base de données
- Recommandations basées sur la sémantique

### 4️⃣ **Découverte CLIP** 🔍
- Recherche par **description textuelle** (ex: "un film avec des robots")
- Modèle **CLIP** (vision + texte alignés)
- Recherche multi-modale dans la base de films

---

## 🚀 Installation & Déploiement

### Prérequis
- Docker & Docker Compose installés

### Étapes

#### 1. **Cloner le repository**
```bash
git clone <ton-repo>
cd projet_AIF
```

#### 2. **Lancer les containers**
```bash
docker-compose up --build
```

L'interface Gradio sera accessible à:
```
http://localhost:7860
```


### ⚠️ Arrêter les containers
```bash
docker-compose down
```

---

## 📁 Structure du Projet

```
projet_AIF/
├── app_gradio.py                    # Interface Gradio principale
├── annoy_db.py                      # API Flask pour recommandations Annoy
├── movieposter_api.py               # API Flask pour classification
├── build_clip_index.py              # Script de construction de l'index CLIP
├── annoy_clip.py                    # API Flask pour découverte CLIP
│
├── Dockerfile.gradio                # Container Gradio
├── Dockerfile.annoy                 # Container API Annoy
├── Dockerfile.api                   # Container API Classification
├── Dockerfile.annoy_clip            # Container CLIP
├── docker-compose.yml               # Orchestration des containers
│
├── requirements-gradio.txt          # Dépendances Gradio
├── requirements-annoy.txt           # Dépendances Annoy
├── requirements-api.txt             # Dépendances API
├── requirements-annoy-clip.txt      # Dépendances CLIP
│
├── .gitignore                       # Fichiers à ignorer
└── README.md                        # Ce fichier

```

---

## ⚙️ Configuration

### Variables d'Environnement

Dans `docker-compose.yml`:

```yaml
environment:
  API_CLASSIFICATION_URL: "http://localhost:5075/predict"
  API_RECOMMENDATION_URL: "http://localhost:5076/recommend"
  ANNOY_URL: "http://annoy:5000/recommend"
  ANNOY_CLIP_URL: "http://annoy_clip:5077"
```

### Ports utilisés

| Service | Port | Rôle |
|---------|------|------|
| Gradio | 7860 | Interface web |
| Annoy | 5000 | Recommandations images |
| Classification | 5075 | Prédiction genre |
| Movie API | 5076 | Métadonnées films |
| CLIP | 5077 | Découverte multi-modale |

---

## 🔑 Fichiers Clés

### `app_gradio.py`
**Rôle**: Interface utilisateur principale
- 4 onglets: Classification | Recommandation | Synopsis | CLIP
- Extraction de features avec MobileNet v3 Small
- Appels aux APIs Flask
- Rendu HTML des résultats

---

### `annoy_db.py`
**Rôle**: API Flask pour recommandations par similarité
- Charge l'index Annoy pré-construit
- Recherche des K plus proches voisins
- Sert les images via route `/posters`

---

### `movieposter_api.py`
**Rôle**: API Flask pour classification de genre
- Modèle CNN pré-entrainé
- Prédiction du genre principal d'un poster

---

### `build_clip_index.py`
**Rôle**: Script de construction de l'index CLIP
- Extrait les embeddings CLIP de tous les posters
- Construit un index Annoy pour CLIP
- Sauvegarde les embeddings

---

### `annoy_clip.py`
**Rôle**: API Flask pour découverte multi-modale CLIP
- Recherche par texte (ex: "sci-fi robots")
- Alignement texte-image via CLIP

---

## 👥 Auteurs

- Projet d'étude - INSA Toulouse
- Cours: AIF
- Albane Vergnes, Julie Laurent, Ema Galuppini

---
