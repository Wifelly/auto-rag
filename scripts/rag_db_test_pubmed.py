import argparse
import hashlib
import json
import os
import pickle
import traceback

from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

console = Console()
HASHES_FILE = "file_hashes.json"


class TqdmEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts, batch_size=10):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Создание эмбеддингов", unit="батч"):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_query(self, text):
        return self.model.encode([text])[0]


def calculate_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def load_hashes():
    if os.path.exists(HASHES_FILE):
        with open(HASHES_FILE) as f:
            return json.load(f)
    return {}


def save_hashes(hashes):
    with open(HASHES_FILE, "w") as f:
        json.dump(hashes, f, indent=2)


def extract_text_from_txt_files(folder_path, max_files=None):
    if not os.path.exists(folder_path):
        console.print(f"[bold red]Папка не найдена:[/bold red] {folder_path}")
        return ""

    txt_files = [
        f for f in os.listdir(folder_path) if f.endswith(".txt") and os.path.isfile(os.path.join(folder_path, f))
    ]

    if max_files and len(txt_files) > max_files:
        console.print(f"[yellow]Ограничение:[/yellow] будет обработано {max_files} из {len(txt_files)} файлов")
        txt_files = txt_files[:max_files]

    if not txt_files:
        console.print("[yellow]В папке нет .txt файлов[/yellow]")
        return ""

    console.print(f"[cyan]Обработка {len(txt_files)} .txt файлов из {folder_path}[/cyan]")
    full_text = ""
    current_hashes = {}
    previous_hashes = load_hashes()

    for filename in tqdm(txt_files, desc="Чтение файлов", unit="файл"):
        file_path = os.path.join(folder_path, filename)
        file_hash = calculate_file_hash(file_path)
        current_hashes[filename] = file_hash

        if filename in previous_hashes and previous_hashes[filename] == file_hash:
            console.log(f"Пропуск {filename} — не изменялся.")
            continue

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                full_text += f.read() + "\n\n"
            console.log(f"Обработан: {filename}")
        except Exception as e:
            console.print(f"[red]Ошибка при чтении {file_path}:[/red] {e}")

    save_hashes(current_hashes)
    return full_text


def create_vector_db(
    text,
    output_dir="pubmed_db_index",
    chunk_size=1024,
    chunk_overlap=200,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    append=False,
):
    if not text.strip():
        console.print("[red]Нет нового текста для обработки[/red]")
        return False

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    console.print("[blue]Разделение текста на чанки...[/blue]")
    chunks = splitter.split_text(text)
    console.print(f"[green]Чанков создано:[/green] {len(chunks)}")

    documents = []
    metadocs = []

    for i, chunk in enumerate(tqdm(chunks, desc="Подготовка документов", unit="чанк")):
        documents.append(Document(page_content=chunk, metadata={"source": "pubmed_txt", "chunk_id": i}))
        metadocs.append({"title": f"PubMed Chunk {i}", "text": chunk, "source": "pubmed_txt"})

    if not documents:
        console.print("[red]Не удалось создать документы для индексации[/red]")
        return False

    try:
        console.print(f"[bold]Загрузка модели эмбеддингов:[/bold] {model_name}")
        embeddings = TqdmEmbeddings(model_name)

        os.makedirs(output_dir, exist_ok=True)
        index_path = os.path.join(output_dir, "index.pkl")
        faiss_path = os.path.join(output_dir, "index.faiss")

        if append and os.path.exists(faiss_path):
            console.print("[yellow]Загрузка существующего индекса FAISS...[/yellow]")
            db = FAISS.load_local(output_dir, embeddings)
        else:
            console.print("[bold blue]Создание новой базы FAISS...[/bold blue]")
            db = FAISS.from_documents(documents, embeddings)
            with open(index_path, "wb") as f:
                pickle.dump(metadocs, f)
            db.save_local(output_dir)
            console.print(f"[bold green]База создана в: {output_dir}[/bold green]")
            return True

        console.print("[green]Добавление новых документов в индекс...[/green]")
        db.add_documents(documents)
        db.save_local(output_dir)

        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                old_metadocs = pickle.load(f)
        else:
            old_metadocs = []

        all_metadocs = old_metadocs + metadocs
        with open(index_path, "wb") as f:
            pickle.dump(all_metadocs, f)

        console.print(f"[bold green]Индекс обновлён в: {output_dir}[/bold green]")
        return True

    except Exception as e:
        console.print(f"[bold red]Ошибка при создании/обновлении индекса:[/bold red] {e}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Создание или обновление FAISS-векторной базы из текстов PubMed")
    parser.add_argument("--input-dir", default="pubmed_txt", help="Папка с .txt файлами")
    parser.add_argument("--output-dir", default="pubmed_db_index", help="Папка для сохранения индекса")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Размер чанка в символах")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Перекрытие между чанками")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Модель эмбеддингов",
    )
    parser.add_argument("--max-files", type=int, help="Ограничить количество файлов")
    parser.add_argument("--append", action="store_true", help="Добавить к существующему индексу")

    args = parser.parse_args()

    console.rule("[bold blue]Запуск обработки PubMed текстов")

    text = extract_text_from_txt_files(args.input_dir, args.max_files)

    if text:
        console.print(f"Объём нового текста: {len(text)} символов (~{len(text.split())} слов)")
        create_vector_db(
            text=text,
            output_dir=args.output_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            model_name=args.model,
            append=args.append,
        )
    else:
        console.print("[bold yellow]Нечего индексировать. Все файлы уже обработаны.[/bold yellow]")


if __name__ == "__main__":
    main()
