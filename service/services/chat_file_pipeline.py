from fastapi import UploadFile
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from service.monitoring.logger import logger
from service.services.utils import Utils

MAX_TEXT_SIZE = 50_000_000
MAX_ALLOWED_CHUNKS = 50000


class ChatFileHandler:
    def __init__(self):
        self.utils = Utils()

    async def handle_uploaded_file(self, file, chunk_size: int = 500, chunk_overlap: int = 100):
        try:
            if not isinstance(file, UploadFile):
                logger.warning(f"Ожидался объект типа UploadFile, но получен {type(file)}. Пропуск.")
                return None

            if not file.content_type.startswith("text") and file.content_type != "application/pdf":
                logger.warning(f"Файл {file.filename} не является текстовым или PDF. Пропуск.")
                return None

            logger.info(f"Начинаю обработку файла {file.filename}.")

            content = await file.read()
            text = await self.utils.extract_text_from_bytes(file.filename, content)

            if not text.strip():
                logger.warning(f"Файл {file.filename} не содержит текста.")
                return None

            text = self.utils.clean_text(text)

            if len(text.encode("utf-8")) > MAX_TEXT_SIZE:
                logger.warning(f"Текст в файле {file.filename} превышает максимальный размер. Обрезка...")
                text = text[:MAX_TEXT_SIZE]

            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " "],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = splitter.split_text(text)

            if len(chunks) > MAX_ALLOWED_CHUNKS:
                logger.warning(f"Файл {file.filename} дал слишком много чанков ({len(chunks)}).")
                return None

            documents = [
                Document(page_content=chunk, metadata={"source": file.filename, "chunk_id": i})
                for i, chunk in enumerate(chunks)
                if chunk.strip()
            ]

            logger.info(f"Файл {file.filename} успешно обработан. Создано {len(documents)} документа(-ов).")

            return documents

        except Exception as e:
            logger.error(f"Ошибка при обработке файла {file.filename}: {e}")
            return None
