import asyncio
from datetime import UTC, datetime
from pathlib import Path

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models.embedding.embedding import DEFAULT_MODEL_NAME, EmbeddingModel
from sqlalchemy.ext.asyncio import AsyncSession

from service.monitoring.logger import logger
from service.services.embedding_service import EmbeddingService
from service.services.utils import Utils

MAX_CONCURRENT_FILE_READS = 10
MAX_ALLOWED_CHUNKS = 50000


class PubmedPipeline:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        knowledge_base_link: str = "pubmed_txt",
        hashes_file: str = "file_hashes_pubmed.json",
    ) -> None:
        self.embeddings: EmbeddingModel | None = None
        self.model_name = model_name
        self.knowledge_base_link = knowledge_base_link
        self.utils = Utils(hashes_file=hashes_file)

    async def _load_embeddings(self) -> None:
        if self.embeddings is None:
            logger.info(f"Lazy loading embedding model: {self.model_name}")
            self.embeddings = EmbeddingModel(model_name=self.model_name)

    async def extract_texts_with_hash_check(
        self,
        folder_path: str,
        max_files: int | None = None,
    ) -> tuple[str, dict[str, str], list[str]]:
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder '{folder_path}' not found.")

        files = [f for f in folder.iterdir() if f.suffix == ".txt" and f.is_file()]
        if not files:
            logger.warning("No .txt files found for processing.")
            return "", {}, []

        if max_files is not None:
            files = files[:max_files]

        previous_hashes = self.utils.load_hashes()
        current_hashes: dict[str, str] = {}
        extracted_filenames: list[str] = []
        tasks: list[asyncio.Task[str]] = []

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_FILE_READS)

        async def read_file_with_limit(file_path: Path) -> str:
            async with semaphore:
                return await self.utils.async_read_txt_file(file_path)

        for file_path in files:
            file_hash = self.utils.calculate_file_hash(file_path)
            current_hashes[file_path.name] = file_hash

            if previous_hashes.get(file_path.name) == file_hash:
                logger.info(f"Skipped {file_path.name} (no changes)")
            else:
                tasks.append(asyncio.create_task(read_file_with_limit(file_path)))
                extracted_filenames.append(file_path.name)

        if not tasks:
            logger.warning("No new or modified files for processing.")
            return "", {}, []

        results = await asyncio.gather(*tasks)
        combined_text = "\n\n".join(results)
        return combined_text, current_hashes, extracted_filenames

    async def run_pipeline(
        self,
        pubmed_dir: str,
        output_dir: str,
        db: AsyncSession | None = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        max_files: int | None = None,
        append: bool = False,
    ) -> bool:
        text, current_hashes, extracted_filenames = await self.extract_texts_with_hash_check(pubmed_dir, max_files)

        if not text:
            logger.warning("No new text available to create database.")
            return False

        success = await self.create_vector_db(text, output_dir, chunk_size, chunk_overlap, append)

        if not success:
            logger.error("Failed to create FAISS index.")
            return False

        if db is not None and extracted_filenames:
            return await self._save_embedding_to_db(db, extracted_filenames, output_dir)

        return True

    async def create_vector_db(
        self,
        text: str,
        output_dir: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        append: bool = False,
    ) -> bool:
        await self._load_embeddings()
        text = self.utils.clean_text(text)

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " "],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_text(text)

        if len(chunks) > MAX_ALLOWED_CHUNKS:
            raise ValueError(f"Too many chunks: {len(chunks)} (>{MAX_ALLOWED_CHUNKS})")

        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": self.knowledge_base_link,
                    "chunk_id": i,
                    "pubmed_id": "PMID:12345678",  # Добавленный PubMed ID
                    "title": "Modern Approaches to Pertussis Infection",  # Название статьи
                    "authors": ["Author1", "Author2", "Author3"],  # Список авторов
                    "doi": "10.1007/xyz123",  # DOI статьи
                    "year": 2023,  # Год публикации
                    "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",  # URL статьи
                },
            )
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

            if not faiss_file.exists():
                logger.error(f"FAISS index was not saved: {faiss_file}")
                return False

            logger.info(f"Vector database successfully saved to: {faiss_file}")
            return True

        except Exception as e:
            logger.error(f"Error creating/updating FAISS index: {e}")
            return False

    async def _save_embedding_to_db(
        self,
        db: AsyncSession,
        filenames: list[str],
        output_dir: str,
    ) -> bool:
        try:
            embedding_service = EmbeddingService(db)
            embedding = await embedding_service.create_embedding(
                name=f"pubmed_embedding_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}",
                files=filenames,
                status_id=1,
                vector_db_path=str(Path(output_dir) / "index.faiss"),
            )
            if embedding is None:
                logger.error("Failed to save embedding to the database.")
                return False
            return True
        except Exception as e:
            logger.error(f"Error saving embedding to the database: {e}")
            return False
