from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from starlette.datastructures import UploadFile

from service.monitoring.logger import logger
from service.services.utils import Utils

MAX_TEXT_SIZE = 50_000_000
MAX_ALLOWED_CHUNKS = 50_000


class ChatFileHandler:
    def __init__(self):
        self.utils = Utils()

    async def handle_uploaded_file(self, file: UploadFile, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Принимает starlette UploadFile, извлекает из него текст, чистит,
        разбивает на чанки и возвращает список Document для дальнейшей обработки.
        """
        try:
            if not isinstance(file, UploadFile):
                logger.warning(f"Ожидался UploadFile, но получен {type(file)} — пропуск.")
                return None

            ctype = file.content_type or ""
            if not ctype.startswith("text") and ctype not in ("application/pdf", "application/octet-stream"):
                logger.warning(f"Файл {file.filename} с content_type={ctype} не поддерживается — пропуск.")
                return None

            logger.info(f"Начинаю обработку файла {file.filename} (content_type={ctype}).")

            content = await file.read()

            text = await self.utils.extract_text_from_bytes(file.filename, content)
            if not text or not text.strip():
                logger.warning(f"Файл {file.filename} не содержит извлекаемого текста.")
                return None

            text = self.utils.clean_text(text)

            if len(text.encode("utf-8")) > MAX_TEXT_SIZE:
                logger.warning(f"Текст в файле {file.filename} превышает {MAX_TEXT_SIZE} байт — обрезаю.")
                text = text[:MAX_TEXT_SIZE]

            splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", " "],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = splitter.split_text(text)
            if len(chunks) > MAX_ALLOWED_CHUNKS:
                logger.warning(f"Файл {file.filename} дал {len(chunks)} чанков (> {MAX_ALLOWED_CHUNKS}) — пропуск.")
                return None

            documents = [
                Document(page_content=chunk, metadata={"source": file.filename, "chunk_id": i})
                for i, chunk in enumerate(chunks)
                if chunk.strip()
            ]
            logger.info(f"Файл {file.filename} успешно обработан: {len(documents)} документов.")
            return documents

        except Exception:
            logger.exception(f"Критическая ошибка при обработке файла {file.filename}")
            return None
