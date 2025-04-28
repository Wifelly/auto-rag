# --- embeddings.py ---

from typing import List, Optional
import torch
from langchain.embeddings.base import Embeddings as BaseEmbeddings
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL_NAME: str = "intfloat/multilingual-e5-large"


class EmbeddingModel(BaseEmbeddings):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None) -> None:
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели эмбеддингов '{model_name}'") from e

    def embed_documents(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Эмбеддинг батчей", unit="батч"):
            batch: List[str] = texts[i : i + batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                model_output = self.model(**inputs)
            embeddings = self._mean_pooling(model_output, inputs["attention_mask"])
            all_embeddings.extend(embeddings.cpu().numpy())
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    @staticmethod
    def _mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
