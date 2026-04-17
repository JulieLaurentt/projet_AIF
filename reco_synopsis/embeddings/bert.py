import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from tqdm import tqdm


class PlotDataset(Dataset):
    """Dataset custom comme dans le TP, adapté pour les synopsis."""
    def __init__(self, texts: list[str], tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512  # synopsis peuvent être longs
        )

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])


class BERTEmbedder:
    """
    DistilBERT embedder — fidèle au TP :
    - tokenisation avec DistilBertTokenizerFast
    - embedding = hidden state du token [CLS] de la dernière couche (index 0)
    - modèle freezé, pas de fine-tuning (on veut juste les embeddings)
    """
    def __init__(self, model_name: str = 'distilbert-base-uncased', batch_size: int = 16):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"BERT utilise : {self.device}")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertModel.from_pretrained(
            model_name,
            output_hidden_states=True  # nécessaire pour récupérer les hidden states comme dans le TP
        ).to(self.device)

        # Freeze tous les paramètres — on veut juste les embeddings
        for param in self.model.parameters():
            param.requires_grad = False

        self.batch_size = batch_size
        self.embeddings = None

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        Extrait les embeddings CLS exactement comme dans le TP :
        emb[-1][:, 0, :] = dernière couche, token CLS (position 0)
        """
        dataset = PlotDataset(texts, self.tokenizer)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        all_embeddings = []
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(loader, desc="Encodage BERT"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                output = self.model(input_ids, attention_mask=attention_mask)

                # Exactement comme le TP : hidden_states[-1][:, 0, :]
                hidden_states = output.hidden_states
                cls_embeddings = hidden_states[-1][:, 0, :]  # token [CLS]

                all_embeddings.append(cls_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def fit_transform(self, plots: list[str]) -> np.ndarray:
        plots = [p if isinstance(p, str) else '' for p in plots]
        self.embeddings = self._get_embeddings(plots)
        return self.embeddings

    def transform(self, texts: list[str]) -> np.ndarray:
        return self._get_embeddings(texts)

    def save(self, path: str = "saved_models/bert"):
        import os
        os.makedirs(path, exist_ok=True)
        np.save(f"{path}/embeddings.npy", self.embeddings)

    def load(self, path: str = "saved_models/bert"):
        self.embeddings = np.load(f"{path}/embeddings.npy")