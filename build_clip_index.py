
import argparse, os, json, base64, io, torch, numpy as np
from PIL import Image
from tqdm import tqdm
from annoy import AnnoyIndex
from transformers import CLIPModel, CLIPProcessor
import pandas as pd

MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_DIM   = 512
N_TREES    = 50

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--size",        type=int, default=1000)
    p.add_argument("--output_dir",  type=str, default="weights/")
    p.add_argument("--csv_path",    type=str, default="data/movie_plots.csv")
    p.add_argument("--posters_dir", type=str, default=".")
    return p.parse_args()

def get_text_embedding(model, processor, text, device):
    inputs = processor(text=[text], return_tensors="pt", padding=True,
                       truncation=True, max_length=77).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy().astype(np.float32)

def get_image_embedding(model, processor, image, device):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu().numpy().astype(np.float32)

def image_to_base64(image, size=(150, 220)):
    image = image.resize(size)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Chargement CLIP ({MODEL_NAME})...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model     = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    print(f"Chargement CSV : {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    df = df.dropna(subset=["movie_plot", "movie_poster_path"])
    df = df.head(args.size)
    print(f"Indexation de {len(df)} films...")

    index_posters = AnnoyIndex(CLIP_DIM, 'angular')
    index_plots   = AnnoyIndex(CLIP_DIM, 'angular')
    metadata = []

    valid_idx = 0
    skipped   = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            poster_path = os.path.join(args.posters_dir, row["movie_poster_path"])
            plot        = str(row["movie_plot"]).strip()
            category    = str(row.get("movie_category", ""))

            if not os.path.exists(poster_path):
                skipped += 1
                continue

            img     = Image.open(poster_path).convert("RGB")
            img_emb = get_image_embedding(model, processor, img, device)
            txt_emb = get_text_embedding(model, processor, plot[:500], device)

            index_posters.add_item(valid_idx, img_emb.tolist())
            index_plots.add_item(valid_idx, txt_emb.tolist())

            img_b64 = image_to_base64(img)
            metadata.append({
                "idx":       valid_idx,
                "title":     category + " — " + os.path.basename(poster_path),
                "plot":      plot[:500],
                "category":  category,
                "image_b64": img_b64,
            })
            valid_idx += 1

        except Exception as e:
            skipped += 1
            continue

    print(f"\n{valid_idx} films indexés, {skipped} ignorés.")

    index_posters.build(N_TREES)
    index_plots.build(N_TREES)
    index_posters.save(os.path.join(args.output_dir, "clip_posters.ann"))
    index_plots.save(os.path.join(args.output_dir, "clip_plots.ann"))
    with open(os.path.join(args.output_dir, "clip_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)

    print(f"✅ Index sauvegardés dans {args.output_dir}")

if __name__ == "__main__":
    main()