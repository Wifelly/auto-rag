import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup

# Константы
BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
UPDATE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
DOWNLOAD_DIR = "pubmed_downloads"


def get_file_list(url):
    """Получает список файлов с FTP-веб-страницы."""
    print(f"Получаем список файлов из {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Используем BeautifulSoup для более надежного парсинга HTML
        soup = BeautifulSoup(response.text, "html.parser")
        files = []

        # Ищем ссылки на .gz файлы
        for link in soup.find_all("a"):
            href = link.get("href")
            if href and href.endswith(".gz"):
                files.append(href)

        return files
    except Exception as e:
        print(f"Ошибка при получении списка файлов: {e}")
        return []


def download_file(url, filepath):
    """Скачивает файл по URL и сохраняет его по указанному пути."""
    try:
        print(f"Скачиваем {os.path.basename(filepath)}...")
        start_time = time.time()

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            downloaded = 0

            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Показываем прогресс для больших файлов
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            if percent % 10 < 0.1:  # Показываем примерно каждые 10%
                                print(f"  Прогресс: {percent:.1f}%")

        elapsed = time.time() - start_time
        print(f"  Завершено за {elapsed:.1f} секунд ({os.path.getsize(filepath) / 1024 / 1024:.1f} МБ)")
        return True
    except Exception as e:
        print(f"Ошибка при скачивании {os.path.basename(filepath)}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)  # Удаляем частично скачанный файл
        return False


def main():
    parser = argparse.ArgumentParser(description="Скачивание файлов PubMed")
    parser.add_argument(
        "--baseline",
        type=int,
        default=5,
        help="Количество файлов baseline для скачивания (по умолчанию: 5)",
    )
    parser.add_argument(
        "--update",
        type=int,
        default=5,
        help="Количество файлов update для скачивания (по умолчанию: 5)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=3,
        help="Количество потоков для параллельного скачивания (по умолчанию: 3)",
    )
    args = parser.parse_args()

    # Создаем директорию для скачивания, если она не существует
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    # Получаем список файлов
    baseline_files = get_file_list(BASELINE_URL)
    update_files = get_file_list(UPDATE_URL)

    print(f"\nНайдено {len(baseline_files)} baseline файлов и {len(update_files)} update файлов.")

    # Выбираем нужное количество файлов
    baseline_files = baseline_files[: args.baseline]
    update_files = update_files[: args.update]

    print(f"Будет скачано {len(baseline_files)} baseline файлов и {len(update_files)} update файлов.")

    # Скачиваем файлы в параллельных потоках
    download_tasks = []

    # Добавляем задачи для baseline файлов
    for filename in baseline_files:
        file_url = BASELINE_URL + filename
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        download_tasks.append((file_url, filepath))

    # Добавляем задачи для update файлов
    for filename in update_files:
        file_url = UPDATE_URL + filename
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        download_tasks.append((file_url, filepath))

    # Запускаем параллельное скачивание
    success_count = 0
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        results = list(executor.map(lambda task: download_file(task[0], task[1]), download_tasks))
        success_count = sum(results)

    print(f"\nЗавершено! Успешно скачано {success_count} из {len(download_tasks)} файлов.")
    print(f"Все файлы сохранены в папке: {os.path.abspath(DOWNLOAD_DIR)}")


if __name__ == "__main__":
    main()
