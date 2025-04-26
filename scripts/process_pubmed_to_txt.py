import argparse
import gzip
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

# Константы
DEFAULT_INPUT_DIR = "pubmed_downloads"
DEFAULT_OUTPUT_DIR = "pubmed_txt"


def extract_text_from_pubmed_xml_gz(gz_path, output_folder):
    """Извлекает текст из сжатого XML-файла PubMed и сохраняет в текстовый файл."""
    try:
        filename = os.path.basename(gz_path)
        base_filename = os.path.splitext(os.path.splitext(filename)[0])[0]
        output_file = os.path.join(output_folder, f"{base_filename}.txt")

        print(f"Обработка: {filename}")
        start_time = time.time()

        # Открываем выходной файл для записи
        with open(output_file, "w", encoding="utf-8") as out_file:
            # Открываем и итеративно обрабатываем XML
            with gzip.open(gz_path, "rb") as f:
                context = ET.iterparse(f, events=("start", "end"))

                # Счетчики для статистики
                article_count = 0

                # Переменные для текущей статьи
                current_article = {}
                in_article = False
                in_abstract = False
                abstract_text = []

                for event, elem in context:
                    tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

                    # Начало новой статьи
                    if event == "start" and tag == "PubmedArticle":
                        in_article = True
                        current_article = {
                            "title": "",
                            "abstract": "",
                            "pmid": "",
                            "keywords": [],
                        }

                    # Обработка PMID
                    elif in_article and event == "end" and tag == "PMID":
                        current_article["pmid"] = elem.text

                    # Заголовок статьи
                    elif in_article and event == "end" and tag == "ArticleTitle":
                        if elem.text:
                            current_article["title"] = elem.text

                    # Начало абстракта
                    elif in_article and event == "start" and tag == "Abstract":
                        in_abstract = True

                    # Текст абстракта (может быть с секциями)
                    elif (
                        in_article
                        and in_abstract
                        and event == "end"
                        and tag == "AbstractText"
                    ):
                        if elem.text:
                            label = elem.get("Label", "")
                            if label:
                                abstract_text.append(f"{label}: {elem.text}")
                            else:
                                abstract_text.append(elem.text)

                    # Конец абстракта
                    elif in_article and event == "end" and tag == "Abstract":
                        in_abstract = False
                        current_article["abstract"] = "\n".join(abstract_text)
                        abstract_text = []

                    # Ключевые слова
                    elif in_article and event == "end" and tag == "Keyword":
                        if elem.text:
                            current_article["keywords"].append(elem.text)

                    # Конец статьи - записываем в файл
                    elif event == "end" and tag == "PubmedArticle":
                        in_article = False

                        # Формируем финальный текст статьи
                        article_text = f"PMID: {current_article['pmid']}\n"
                        article_text += f"TITLE: {current_article['title']}\n"

                        if current_article["abstract"]:
                            article_text += f"ABSTRACT: {current_article['abstract']}\n"

                        if current_article["keywords"]:
                            article_text += (
                                f"KEYWORDS: {', '.join(current_article['keywords'])}\n"
                            )

                        # Записываем статью в файл
                        out_file.write(article_text + "\n\n")

                        # Увеличиваем счетчик
                        article_count += 1

                        # Очищаем элемент для экономии памяти
                        elem.clear()

        elapsed = time.time() - start_time
        print(f"  Завершено за {elapsed:.1f} сек. Обработано статей: {article_count}")
        return True, article_count

    except Exception as e:
        print(f"Ошибка при обработке файла {gz_path}: {e}")
        return False, 0


def process_pubmed_files(input_dir, output_dir, max_workers=4):
    """Обрабатывает все .xml.gz файлы в указанной директории."""
    # Создаем выходную директорию, если она не существует
    os.makedirs(output_dir, exist_ok=True)

    # Получаем список всех .xml.gz файлов
    gz_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".xml.gz")
    ]

    if not gz_files:
        print(f"В директории {input_dir} не найдены .xml.gz файлы.")
        return

    print(f"Найдено {len(gz_files)} файлов для обработки.")

    # Создаем пул потоков для параллельной обработки
    total_articles = 0
    successful_files = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Запускаем обработку всех файлов
        futures = [
            executor.submit(extract_text_from_pubmed_xml_gz, gz_file, output_dir)
            for gz_file in gz_files
        ]

        # Собираем результаты
        for future in futures:
            success, article_count = future.result()
            if success:
                successful_files += 1
                total_articles += article_count

    print("\nОбработка завершена!")
    print(f"Успешно обработано файлов: {successful_files} из {len(gz_files)}")
    print(f"Всего извлечено статей: {total_articles}")
    print(f"Результаты сохранены в директории: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Обработка PubMed XML.GZ файлов в текстовый формат"
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Директория с .xml.gz файлами (по умолчанию: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Директория для сохранения текстовых файлов (по умолчанию: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Количество параллельных потоков обработки (по умолчанию: 4)",
    )
    args = parser.parse_args()

    process_pubmed_files(args.input_dir, args.output_dir, args.threads)


if __name__ == "__main__":
    main()
