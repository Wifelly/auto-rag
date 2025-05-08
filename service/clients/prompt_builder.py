# --- prompt_builder.py ---


class PromptBuilder:
    SYSTEM_PROMPT = "Ты — медицинский помощник."

    @staticmethod
    def build_initial_prompt(query: str) -> list[dict]:
        return [
            {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
            {"role": "user", "content": f"Ответь на вопрос пользователя: {query}"},
        ]

    @staticmethod
    def build_with_context(initial: str, query: str, context: str) -> list[dict]:
        return [
            {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
            {"role": "user", "content": f"Ответь на вопрос пользователя: {query}"},
            {"role": "assistant", "content": initial},
            {
                "role": "user",
                "content": (
                    f"Дополнительно используй эти документы:\n<context>{context}</context>\n"
                    f"Повтори или уточни ответ на вопрос: {query}"
                ),
            },
        ]

    @staticmethod
    def format_prompt(messages: list[dict]) -> str:
        parts = []
        for msg in messages:
            role, content = msg["role"], msg["content"]
            if role == "system":
                parts.append(f"[SYSTEM]\n{content}")
            elif role == "user":
                parts.append(f"[USER]\n{content}")
            elif role == "assistant":
                parts.append(f"[ASSISTANT]\n{content}")
        return "\n\n".join(parts)
