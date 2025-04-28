# --- pubmed_pipeline.py ---

import asyncio
from pathlib import Path

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models.embedding.embedding import DEFAULT_MODEL_NAME, EmbeddingModel
from parsers.pubmed_xml_parser import PubmedXMLParser

from service.service.logger import logger
from service.service.utils import MAX_TEXT_SIZE, Utils


class PubmedPipeline:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        knowledge_base_link: str = "pubmed_txt",
        hashes_file: str = "file_hashes_pubmed.json",
    ):
        self.embeddings = EmbeddingModel(model_name=model_name)
        self.knowledge_base_link = knowledge_base_link
        self.utils = Utils(hashes_file=hashes_file)

    async def extract_texts_with_hash_check(self, folder_path: str, max_files: int | None = None) -> tuple[str, dict]:
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Папка '{folder_path}' не найдена.")

        files = [f for f in folder.iterdir() if f.suffix == ".txt" and f.is_file()]

        if not files:
            logger.warning("Нет .txt файлов для обработки.")
            return "", {}

        if max_files:
            files = files[:max_files]

        previous_hashes = self.utils.load_hashes()
        current_hashes = {}
        tasks = []

        for file_path in files:
            file_hash = self.utils.calculate_file_hash(str(file_path))
            current_hashes[file_path.name] = file_hash

            if previous_hashes.get(file_path.name) == file_hash:
                logger.info(f"Пропущено {file_path.name} (без изменений)")
            else:
                tasks.append(self.utils.async_read_txt_file(str(file_path)))

        if not tasks:
            logger.warning("Нет новых или изменённых файлов для обработки.")
            return "", {}

        results = await asyncio.gather(*tasks)
        return "\n\n".join(results), current_hashes

    async def run_pipeline(
        self,
        pubmed_dir: str,
        output_dir: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        max_files: int | None = None,
        append: bool = False,
    ) -> bool:
        text, current_hashes = await self.extract_texts_with_hash_check(pubmed_dir, max_files)

        if not text:
            logger.warning("Нет нового текста для создания базы.")
            return False

        if len(text.encode("utf-8")) > MAX_TEXT_SIZE:
            raise ValueError("Размер общего текста превышает допустимый лимит")

        success = await self.create_vector_db(text, output_dir, chunk_size, chunk_overlap, append)

        if success:
            self.utils.save_hashes(current_hashes)

        return success

    async def run_pipeline_from_xml(
        self, xml_content: str, output_dir: str, chunk_size: int = 1024, chunk_overlap: int = 200, append: bool = False
    ) -> bool:
        """Обработка XML-контента напрямую без сохранения на диск."""
        if len(xml_content.encode("utf-8")) > MAX_TEXT_SIZE:
            raise ValueError("Размер XML контента превышает допустимый лимит")

        text = PubmedXMLParser.parse_xml_to_text(xml_content)

        if not text.strip():
            logger.warning("Парсинг XML не дал текста для индексации.")
            return False

        return await self.create_vector_db(text, output_dir, chunk_size, chunk_overlap, append)

    async def create_vector_db(
        self, text: str, output_dir: str, chunk_size: int = 1024, chunk_overlap: int = 200, append: bool = False
    ) -> bool:
        text = self.utils.clean_text(text)

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "], chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        documents = [
            Document(page_content=chunk, metadata={"source": self.knowledge_base_link, "chunk_id": i})
            for i, chunk in enumerate(chunks)
            if chunk.strip()
        ]

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        faiss_file = output_path / "index.faiss"

        try:
            if append and faiss_file.exists():
                db = FAISS.load_local(str(output_path), self.embeddings, allow_dangerous_deserialization=True)
                db.add_documents(documents)
            else:
                db = FAISS.from_documents(documents, self.embeddings)

            db.save_local(str(output_path))
            logger.info(f"База успешно сохранена в: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Ошибка при создании/обновлении индекса: {e}")
            return False
