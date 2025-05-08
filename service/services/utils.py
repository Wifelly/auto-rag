import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import aiofiles
import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter

from service.monitoring.logger import logger

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".pptx")
MAX_TEXT_SIZE = 50_000_000  # 50MB


class Utils:
    executor = ThreadPoolExecutor()

    @staticmethod
    def clean_text(text: str) -> str:
        text = text.replace("\x00", "")
        lines = text.splitlines()
        cleaned = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]
        return "\n".join(cleaned).strip()

    @staticmethod
    async def extract_text(file_path: Path) -> str:
        ext = file_path.suffix.lower()
        loop = asyncio.get_running_loop()

        if ext == ".pdf":
            text = await loop.run_in_executor(Utils.executor, Utils._extract_pdf_text, file_path)
        elif ext == ".docx":
            text = await loop.run_in_executor(Utils.executor, Utils._extract_docx_text, file_path)
        elif ext == ".pptx":
            text = await loop.run_in_executor(Utils.executor, Utils._extract_pptx_text, file_path)
        else:
            raise ValueError(f"Неподдерживаемое расширение файла: {ext}")

        if len(text) > MAX_TEXT_SIZE:
            logger.warning(f"Файл {file_path.name} превышает MAX_TEXT_SIZE, усечён.")
            return text[:MAX_TEXT_SIZE]

        return text

    @staticmethod
    def _extract_pdf_text(file_path: Path) -> str:
        try:
            from fitz import open as fitz_open

            doc = fitz_open(file_path)
            lines = []
            for page in doc:
                blocks = page.get_text("blocks")
                for b in blocks:
                    block_text = b[4].strip()
                    if block_text:
                        lines.append(block_text)
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[PDF] Ошибка чтения {file_path}: {e}")
            return ""

    @staticmethod
    def _extract_docx_text(file_path: Path) -> str:
        try:
            from docx import Document as DocxDocument

            doc = DocxDocument(file_path)
            lines = []

            for p in doc.paragraphs:
                text = p.text.strip()
                if text:
                    lines.append(text)

            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text:
                        lines.append(row_text)

            return "\n".join(lines)
        except Exception as e:
            logger.error(f"[DOCX] Ошибка чтения {file_path}: {e}")
            return ""

    @staticmethod
    def _extract_pptx_text(file_path: Path) -> str:
        try:
            from pptx import Presentation

            prs = Presentation(file_path)
            return "\n".join(
                shape.text
                for slide in prs.slides
                for shape in slide.shapes
                if hasattr(shape, "text") and shape.text.strip()
            )
        except Exception as e:
            logger.error(f"[PPTX] Ошибка чтения {file_path}: {e}")
            return ""

    @staticmethod
    async def async_read_txt_file(file_path: Path) -> str:
        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return await f.read()
        except Exception as e:
            logger.error(f"[TXT] Ошибка чтения {file_path}: {e}")
            raise OSError(f"Ошибка чтения TXT файла: {file_path}") from e

    @staticmethod
    async def download_xml(url: str) -> str:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                return response.text
        except Exception as e:
            logger.error(f"[XML] Ошибка загрузки по URL {url}: {e}")
            raise

    @staticmethod
    async def extract_text_from_bytes(filename: str, content: bytes) -> str:
        from docx import Document as DocxDocument
        from fitz import open as fitz_open
        from pptx import Presentation

        ext = Path(filename).suffix.lower()

        try:
            if ext == ".docx":
                doc = DocxDocument(BytesIO(content))
                lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
                for table in doc.tables:
                    for row in table.rows:
                        row_text = " | ".join(cell.text.strip() for cell in row.cells)
                        if row_text:
                            lines.append(row_text)
                text = "\n".join(lines)

            elif ext == ".pptx":
                prs = Presentation(BytesIO(content))
                text = "\n".join(
                    shape.text
                    for slide in prs.slides
                    for shape in slide.shapes
                    if hasattr(shape, "text") and shape.text.strip()
                )

            elif ext == ".pdf":
                doc = fitz_open(stream=content, filetype="pdf")
                lines = []
                for page in doc:
                    blocks = page.get_text("blocks")
                    for b in blocks:
                        block_text = b[4].strip()
                        if block_text:
                            lines.append(block_text)
                text = "\n".join(lines)

            elif ext == ".txt":
                text = content.decode("utf-8", errors="ignore")

            else:
                raise ValueError(f"Неподдерживаемый формат файла: {ext}")

            if len(text) > MAX_TEXT_SIZE:
                logger.warning(f"Файл {filename} превышает MAX_TEXT_SIZE, усечён.")
                return text[:MAX_TEXT_SIZE]

            return text

        except Exception as e:
            logger.error(f"[BYTES] Ошибка обработки файла {filename}: {e}")
            return ""

    @staticmethod
    def slugify_name(name: str) -> str:
        name = name.strip().lower()
        name = re.sub(r"[^\w]+", "_", name)
        return name or "unnamed"


class CustomMedicalTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            separators=["\n\n", "\n•", "\n-", "\n*", "\n", ".", "!", "?", " "],
        )

    def split(self, text: str) -> list[str]:
        return self.splitter.split_text(text)


__all__ = ["Utils", "CustomMedicalTextSplitter", "SUPPORTED_EXTENSIONS", "MAX_TEXT_SIZE"]
