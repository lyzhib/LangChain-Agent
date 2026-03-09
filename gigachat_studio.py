import requests
from typing import Optional, List, ClassVar
from langchain_core.language_models.llms import LLM
from config import GIGACHAT_TOKEN


class GigaChatStudioLLM(LLM):

    api_url: ClassVar[str] = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    @property
    def _llm_type(self) -> str:
        return "gigachat_studio"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        headers = {
            "Authorization": f"Bearer {GIGACHAT_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "GigaChat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload
        )

        data = response.json()

        # 调试输出
        print("API RESPONSE:", data)

        # 兼容不同返回格式
        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        if "candidates" in data:
            return data["candidates"][0]["content"]["parts"][0]["text"]

        if "message" in data:
            return data["message"]["content"]

        return str(data)