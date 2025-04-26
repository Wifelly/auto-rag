import argparse
import os

import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Константы
PUBMED_DB_DIR = "pubmed_vectors"  # База PubMed
DOCS_DB_DIR = "docs_db_index"  # База обычных документов
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3

# Получаем API-ключ из переменной окружения или впиши прямо в кавычки
API_KEY = os.getenv(
    "VSE_GPT_API_KEY",
    "",
)

if not API_KEY:
    raise ValueError(
        "API_KEY не задан. Укажи API-ключ в коде или через переменную окружения."
    )


def load_vector_db(db_dir, model_name=DEFAULT_MODEL):
    """Загрузка векторной базы данных."""
    if not os.path.exists(db_dir):
        print(f"Директория {db_dir} не найдена.")
        return None, None

    try:
        print(f"Загрузка векторной базы из {db_dir}...")
        # Инициализируем модель для эмбеддингов
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)

        # Загружаем векторную базу
        db = FAISS.load_local(db_dir, embedding_model)
        doc_count = len(db.index_to_docstore_id)
        print(f"База {db_dir} загружена. Документов: {doc_count}")
        return db, embedding_model
    except Exception as e:
        print(f"Ошибка при загрузке {db_dir}: {e}")
        return None, None


def load_all_vector_dbs(model_name=DEFAULT_MODEL, use_pubmed=True, use_docs=True):
    """Загрузка всех доступных векторных баз."""
    databases = {}
    embedding_model = None

    if use_pubmed:
        pubmed_db, embedding_model = load_vector_db(PUBMED_DB_DIR, model_name)
        if pubmed_db:
            databases["pubmed"] = pubmed_db

    if use_docs:
        docs_db, embedding_model = load_vector_db(DOCS_DB_DIR, model_name)
        if docs_db:
            databases["docs"] = docs_db

    if not databases:
        print("Не удалось загрузить ни одну векторную базу")
        print("Создайте базу с помощью:")
        print("- rag_db_create_pubmed.py для PubMed")
        print("- docs_db_create.py для обычных документов")
        return None, None

    return databases, embedding_model


def search_in_db(query, db, top_k=DEFAULT_TOP_K, source_name=None):
    """Поиск документов в одной базе."""
    if not db:
        return []

    try:
        results = db.similarity_search(query, k=top_k)

        # Помечаем результаты источником, если указан
        if source_name:
            for doc in results:
                if "source_db" not in doc.metadata:
                    doc.metadata["source_db"] = source_name

        return results
    except Exception as e:
        print(f"Ошибка при поиске: {e}")
        return []


def search_similar_docs(query, databases, top_k=DEFAULT_TOP_K):
    """Поиск документов во всех доступных базах."""
    if not databases:
        print("Базы данных не загружены")
        return []

    try:
        print(f"Поиск по запросу: {query}")

        all_results = []
        per_db_results = {}

        # Определяем, сколько результатов брать из каждой базы
        db_count = len(databases)
        results_per_db = max(1, top_k // db_count)

        # Ищем в каждой базе
        for db_name, db in databases.items():
            print(f"  Поиск в базе {db_name}...")
            db_results = search_in_db(query, db, results_per_db, db_name)
            per_db_results[db_name] = db_results
            all_results.extend(db_results)

        # Если нужно больше результатов, чем мы получили
        if len(all_results) < top_k:
            remaining = top_k - len(all_results)
            # Добираем из баз пропорционально их размеру
            for db_name, db in databases.items():
                if remaining <= 0:
                    break

                additional = min(remaining, results_per_db)
                db_results = search_in_db(
                    query, db, results_per_db + additional, db_name
                )
                # Берем только новые результаты (которых нет в per_db_results)
                new_results = db_results[len(per_db_results[db_name]) :]
                all_results.extend(new_results)
                remaining -= len(new_results)

        # Ограничиваем общее количество результатов
        all_results = all_results[:top_k]

        # Статистика по источникам
        source_counts = {}
        for doc in all_results:
            source_db = doc.metadata.get("source_db", "unknown")
            source_counts[source_db] = source_counts.get(source_db, 0) + 1

        source_stats = ", ".join([f"{k}: {v}" for k, v in source_counts.items()])
        print(f"Найдено {len(all_results)} документов ({source_stats})")

        return all_results
    except Exception as e:
        print(f"Ошибка при выполнении поиска: {e}")
        import traceback

        traceback.print_exc()
        return []


def run_gpt_query(system_prompt, user_query, databases, top_k=DEFAULT_TOP_K):
    """Выполнение запроса к GPT на основе похожих документов."""
    # Ищем похожие документы
    docs = search_similar_docs(user_query, databases, top_k)

    if not docs:
        return "Не удалось найти релевантные документы для ответа."

    # Сохраняем информацию об источниках
    sources_info = []

    print("Найдена информация для ответа:")
    for i, doc in enumerate(docs):
        source_db = doc.metadata.get("source_db", "Неизвестно")
        source = doc.metadata.get("source", "Неизвестно")
        chunk_id = doc.metadata.get("chunk_id", "")

        # Сохраняем информацию об источнике
        source_info = {
            "id": i + 1,
            "db": source_db,
            "source": source,
            "chunk_id": chunk_id,
        }
        sources_info.append(source_info)

        print(f"\nИсточник #{i+1}: [{source_db}] {source} (чанк {chunk_id})")
        content_preview = (
            doc.page_content[:150] + "..."
            if len(doc.page_content) > 150
            else doc.page_content
        )
        print(f"Содержание: {content_preview}")

    print("-----")

    # Объединяем содержимое найденных документов
    message_content = "\n\n".join([doc.page_content for doc in docs])

    user_content = (
        f"Ответь на вопрос пользователя, используя информацию из документа ниже.\n"
        f"<doc>{message_content}</doc>\n"
        f"В конце ответа добавь список использованных источников, "
        f"не ссылаясь на их номер в тексте.\n"
        f"Вопрос: {user_query}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Отправляем запрос в VseGPT API
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    payload = {
        "model": "gpt-3.5-turbo",  # или другой поддерживаемый, если требуется
        "messages": messages,
        "temperature": 0.1,
    }

    url = "https://api.vsegpt.ru/v1/chat/completions"

    try:
        response = requests.post(url, headers=headers, json=payload)

        # Проверяем, успешно ли выполнен запрос
        if response.status_code != 200:
            print("Ошибка при обращении к API:")
            print("Статус:", response.status_code)
            print("Ответ:", response.text)
            return f"Ошибка API: {response.status_code} - {response.text}"

        response_data = response.json()

        # Проверяем наличие ожидаемого поля
        if "choices" not in response_data:
            print("Некорректный ответ от API:")
            print(response_data)
            return "Ошибка: некорректный ответ от API"

        gpt_response = response_data["choices"][0]["message"]["content"]

        # Добавляем информацию о реальных источниках
        response_with_sources = (
            gpt_response + "\n\n" + format_sources_info(sources_info)
        )

        return response_with_sources
    except Exception as e:
        print(f"Ошибка при обращении к API: {e}")
        return f"Ошибка: {str(e)}"


def format_sources_info(sources_info):
    """Форматирование информации об источниках для вывода."""
    if not sources_info:
        return ""

    result = "\n" + "=" * 40 + "\n"
    result += "ИНФОРМАЦИЯ ОБ ИСПОЛЬЗОВАННЫХ ИСТОЧНИКАХ:\n"

    for source in sources_info:
        result += f"[{source['id']}] База данных: {source['db']}\n"
        result += f"    Источник: {source['source']}\n"
        if source["chunk_id"]:
            result += f"    Чанк: {source['chunk_id']}\n"
        result += "\n"

    return result


def format_result(doc, index, max_content_length=1000):
    """Форматирование результата поиска для вывода."""
    content = doc.page_content
    if len(content) > max_content_length:
        content = content[:max_content_length] + "..."

    source_db = doc.metadata.get("source_db", "Неизвестно")
    source = doc.metadata.get("source", "Неизвестно")
    chunk_id = doc.metadata.get("chunk_id", "")

    result = f"\n{'='*40}\n"
    result += f"Результат #{index+1} из базы [{source_db}]\n"
    result += f"Источник: {source}\n"
    result += f"Идентификатор чанка: {chunk_id}\n"
    result += f"{'-'*40}\n"
    result += f"{content}\n"
    result += f"{'='*40}\n"

    return result


def main():
    """Основная функция скрипта."""
    parser = argparse.ArgumentParser(
        description="Поиск по векторным базам с возможностью ответов от GPT"
    )
    parser.add_argument("query", type=str, nargs="?", help="Поисковый запрос")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Модель для эмбеддингов (по умолчанию: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Количество результатов для вывода (по умолчанию: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--no-gpt", action="store_true", help="Только поиск без запроса к GPT"
    )
    parser.add_argument(
        "--use-pubmed", action="store_true", help="Использовать только базу PubMed"
    )
    parser.add_argument(
        "--use-docs",
        action="store_true",
        help="Использовать только базу обычных документов",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Интерактивный режим для повторных запросов",
    )

    args = parser.parse_args()

    # Определяем, какие базы использовать
    use_pubmed = args.use_pubmed
    use_docs = args.use_docs

    # Если не указано конкретно, используем обе базы
    if not use_pubmed and not use_docs:
        use_pubmed = True
        use_docs = True

    # Загружаем векторные базы
    databases, _ = load_all_vector_dbs(
        args.model, use_pubmed=use_pubmed, use_docs=use_docs
    )

    if not databases:
        return

    # Система, используемая для GPT
    system_prompt = (
        "Ты — интеллектуальный помощник, отвечающий на вопросы "
        "на основе загруженных документов из разных источников."
    )

    if args.interactive:
        # Интерактивный режим
        while True:
            query = input("\nВведите запрос (или 'выход' для завершения): ")
            if query.lower() in ["выход", "exit", "quit", "q"]:
                break

            if not query.strip():
                continue

            if args.no_gpt:
                # Только поиск без GPT
                results = search_similar_docs(query, databases, args.top_k)

                # Выводим результаты
                if results:
                    for i, doc in enumerate(results):
                        print(format_result(doc, i))
            else:
                # Поиск с использованием GPT
                result = run_gpt_query(system_prompt, query, databases, args.top_k)
                print(f"\nОтвет от GPT:\n{result}")

    elif args.query:
        # Одиночный запрос из аргументов командной строки
        if args.no_gpt:
            # Только поиск без GPT
            results = search_similar_docs(args.query, databases, args.top_k)

            # Выводим результаты
            if results:
                for i, doc in enumerate(results):
                    print(format_result(doc, i))
        else:
            # Поиск с использованием GPT
            result = run_gpt_query(system_prompt, args.query, databases, args.top_k)
            print(f"\nОтвет от GPT:\n{result}")

    else:
        # Нет запроса
        print(
            "\nУкажите поисковый запрос в аргументах или используйте интерактивный режим"
        )
        print("Пример: python rag_db_search.py 'ваш запрос'")
        print("      или python rag_db_search.py --interactive")
        print("\nДоступные флаги:")
        print("  --use-pubmed   - использовать только базу PubMed")
        print("  --use-docs     - использовать только базу документов")
        print("  --no-gpt       - только поиск без запроса к GPT")
        print("  --top-k NUMBER - количество результатов")


if __name__ == "__main__":
    main()
