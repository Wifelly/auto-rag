from service.config import Config


class PromptBuilder:
    SYSTEM_PROMPT = Config.SYSTEM_PROMPT

    @staticmethod
    def build_initial_prompt(query: str) -> list[dict]:
        return [
            {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

    @staticmethod
    def build_with_context(initial: str, query: str, context: str) -> list[dict]:
        system_content = PromptBuilder.SYSTEM_PROMPT
        if context:
            system_content += f"\n\nИспользуй следующие фрагменты:\n{context}"
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{query}\n\n- сформируй ответ в виде **нумерованного списка.**"},
        ]
