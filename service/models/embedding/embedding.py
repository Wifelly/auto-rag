from typing import ClassVar

import torch

# ruff: noqa: N812
import torch.nn.functional as F
from langchain.embeddings.base import Embeddings as BaseEmbeddings
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from service.monitoring.logger import logger

DEFAULT_MODEL_NAME: ClassVar[str] = "intfloat/multilingual-e5-large"


class EmbeddingModel(BaseEmbeddings):
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        normalize_embeddings: bool = True,
        batch_size: int = 10,
    ) -> None:
        super().__init__()
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize_embeddings: bool = normalize_embeddings
        self.batch_size: int = batch_size

        try:
            logger.info(f"Загрузка модели эмбеддингов: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            logger.info(f"Модель успешно загружена на устройство: {self.device}")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели эмбеддингов '{model_name}': {e}")
            raise RuntimeError(f"Ошибка загрузки модели эмбеддингов '{model_name}'") from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning("Список текстов для эмбеддинга пуст.")
            raise ValueError("Список текстов для эмбеддинга пуст.")

        max_length = self.tokenizer.model_max_length
        for text in texts:
            if len(self.tokenizer.tokenize(text)) > max_length:
                logger.warning(f"Текст превышает максимальную длину {max_length} токенов и будет усечён.")

        all_embeddings: list[list[float]] = []
        logger.info(f"Начало обработки {len(texts)} текстов батчами по {self.batch_size}.")

        for i in tqdm(range(0, len(texts), self.batch_size), desc="Эмбеддинг батчей", unit="батч"):
            batch: list[str] = texts[i : i + self.batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                model_output = self.model(**inputs)
            embeddings = self._mean_pooling(model_output, inputs["attention_mask"])

            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.extend(embeddings.cpu().float().numpy())

        logger.info("Обработка текстов завершена.")
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        if not text:
            logger.warning("Текст для эмбеддинга пуст.")
            raise ValueError("Текст для эмбеддинга пуст.")
        return self.embed_documents([text])[0]

    @staticmethod
    def _mean_pooling(
        model_output: BaseModelOutputWithPoolingAndCrossAttentions, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
