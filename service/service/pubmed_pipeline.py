# --- pubmed_pipeline.py ---

import os
import asyncio
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from embeddings import EmbeddingModel, DEFAULT_MODEL_NAME
from utils import Utils, MAX_TEXT_SIZE
from parsers.pubmed_xml_parser import PubmedXMLParser
from logger import logger

class PubmedPipeline:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, knowledge_base_link: str = "pubmed_txt", hashes_file: str = "file_hashes_pubmed.json"):
        self.embeddings = EmbeddingModel(model_name=model_name)
        self.knowledge_base_link = knowledge_base_link
        Utils.hashes_file = hashes_file

    async def extract_texts_with_hash_check(self, folder_path: str, max_files: int = None) -> (str, dict):
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Папка '{folder_path}' не найдена.")

        files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".txt") and os.path.isfile(os.path.join(folder_path, f))
        ]

        if not files:
            logger.warning("Нет .txt файлов для обработки.")
            return "", {}

        if max_files:
            files = files[:max_files]

        previous_hashes = Utils.load_hashes()
        current_hashes = {}
        tasks = []

        for filename in files:
            file_path = os.path.join(folder_path, filename)
            file_hash = Utils.calculate_file_hash(file_path)
            current_hashes[filename] = file_hash

            if previous_hashes.get(filename) == file_hash:
                logger.info(f"Пропущено {filename} (без изменений)")
            else:
                tasks.append(Utils.async_read_txt_file(file_path))

        if not tasks:
            logger.warning("Нет новых или изменённых файлов для обработки.")
            return "", {}

        results = await asyncio.gather(*tasks)
        return "\n\n".join(results), current_hashes

    async def run_pipeline(self, pubmed_dir: str, output_dir: str, chunk_size: int = 1024, chunk_overlap: int = 200, max_files: int = None, append: bool = False) -> bool:
        text, current_hashes = await self.extract_texts_with_hash_check(pubmed_dir, max_files)

        if not text:
            logger.warning("Нет нового текста для создания базы.")
            return False

        if len(text.encode('utf-8')) > MAX_TEXT_SIZE:
            raise ValueError("Размер общего текста превышает допустимый лимит")

        success = await self.create_vector_db(text, output_dir, chunk_size, chunk_overlap, append)

        if success:
            Utils.save_hashes(current_hashes)

        return success

    async def run_pipeline_from_xml(self, xml_content: str, output_dir: str, chunk_size: int = 1024, chunk_overlap: int = 200, append: bool = False) -> bool:
        """Обработка XML-контента напрямую без сохранения на диск."""
        if len(xml_content.encode('utf-8')) > MAX_TEXT_SIZE:
            raise ValueError("Размер XML контента превышает допустимый лимит")

        text = PubmedXMLParser.parse_xml_to_text(xml_content)

        if not text.strip():
            logger.warning("Парсинг XML не дал текста для индексации.")
            return False

        return await self.create_vector_db(text, output_dir, chunk_size, chunk_overlap, append)

    async def create_vector_db(self, text: str, output_dir: str, chunk_size: int = 1024, chunk_overlap: int = 200, append: bool = False) -> bool:
        text = Utils.clean_text(text)

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        documents = [Document(page_content=chunk, metadata={"source": self.knowledge_base_link, "chunk_id": i}) for i, chunk in enumerate(chunks) if chunk.strip()]

        os.makedirs(output_dir, exist_ok=True)
        faiss_path = os.path.join(output_dir, "index.faiss")

        try:
            if append and os.path.exists(faiss_path):
                db = FAISS.load_local(output_dir, self.embeddings, allow_dangerous_deserialization=True)
                db.add_documents(documents)
            else:
                db = FAISS.from_documents(documents, self.embeddings)

            db.save_local(output_dir)
            logger.info(f"База успешно сохранена в: {output_dir}")
            return True

        except Exception as e:
            logger.error(f"Ошибка при создании/обновлении индекса: {e}")
            return False