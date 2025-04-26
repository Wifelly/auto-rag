import os

import requests
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Настройки
load_dotenv("api_keys.env")
API_KEY = ""  # os.getenv("VSE_GPT_API_KEY", "")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3
PUBMED_DB_DIR = "pubmed_vectors"
DOCS_DB_DIR = "docs_db_index"
GPT_URL = "https://api.vsegpt.ru/v1/chat/completions"
GPT_MODEL = "openai/gpt-4o-mini"

# GPT вызов


def call_gpt(messages):
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": GPT_MODEL,
        "messages": messages,
        "temperature": 0.1,
    }
    response = requests.post(GPT_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print("Ошибка API GPT:", response.status_code, response.text)
        return "[Ошибка GPT]"


def is_uncertain(text):
    text = text.lower()
    uncertain_phrases = [
        "не уверен",
        "не могу ответить",
        "неизвестно",
        "недостаточно информации",
        "возможно",
        "предположительно",
        "я не знаю",
        "не знаю",
    ]
    return any(p in text for p in uncertain_phrases)


def search_similar_docs(query, db, top_k):
    return db.similarity_search(query, k=top_k)


def run_gpt_with_fallback(query, dbs, top_k=DEFAULT_TOP_K):
    # Первый GPT-запрос без контекста
    messages = [
        {"role": "system", "content": "Ты — медицинский помощник."},
        {"role": "user", "content": f"Ответь на вопрос пользователя: {query}"},
    ]
    gpt_answer = call_gpt(messages)

    if is_uncertain(gpt_answer):
        print("GPT сомневается. Переходим к RAG...")
        all_docs = []
        for db in dbs.values():
            all_docs.extend(search_similar_docs(query, db, top_k))

        if not all_docs:
            return gpt_answer + "\n\n(Релевантных источников не найдено)"

        context = "\n\n".join([d.page_content for d in all_docs])
        messages.append({"role": "assistant", "content": gpt_answer})
        messages.append(
            {
                "role": "user",
                "content": f"Используй документы ниже, чтобы ответить точнее.\n<context>{context}</context>\nВопрос: {query}",
            }
        )
        refined_answer = call_gpt(messages)
        return refined_answer
    else:
        return gpt_answer


if __name__ == "__main__":
    from sys import argv

    print("Загрузка векторных баз...")
    embedding_model = HuggingFaceEmbeddings(model_name=DEFAULT_MODEL)
    dbs = {}
    if os.path.exists(PUBMED_DB_DIR):
        dbs["pubmed"] = FAISS.load_local(PUBMED_DB_DIR, embedding_model)
    if os.path.exists(DOCS_DB_DIR):
        dbs["docs"] = FAISS.load_local(DOCS_DB_DIR, embedding_model)

    if len(argv) > 1:
        query = " ".join(argv[1:])
    else:
        query = input("Введите вопрос: ")

    answer = run_gpt_with_fallback(query, dbs)
    print("\nОтвет:\n", answer)
