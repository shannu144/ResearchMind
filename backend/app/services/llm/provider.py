import os
import httpx
from abc import ABC, abstractmethod
from typing import Optional
from app.core.config import settings
from app.core.logging import logger


class BaseLLMProvider(ABC):
    """
    Abstract Base Class for LLM Provider Interface.
    """

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass


class OpenAILLMProvider(BaseLLMProvider):
    """
    OpenAI API Provider (GPT-4o / GPT-4o-mini).
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL

    @property
    def provider_name(self) -> str:
        return f"OpenAI ({self.model})"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            logger.warning("OpenAI API key missing. Falling back to Mock LLM Provider.")
            return MockLLMProvider().generate(prompt, system_prompt)

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {"model": self.model, "messages": messages, "temperature": 0.2}

            with httpx.Client(timeout=30.0) as client:
                resp = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    logger.error(f"OpenAI API Error {resp.status_code}: {resp.text}")
                    return MockLLMProvider().generate(prompt, system_prompt)
        except Exception as e:
            logger.error(f"OpenAI Request exception: {e}")
            return MockLLMProvider().generate(prompt, system_prompt)


class GeminiLLMProvider(BaseLLMProvider):
    """
    Google Gemini Provider (Gemini 1.5 Flash).
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL

    @property
    def provider_name(self) -> str:
        return f"Google Gemini ({self.model})"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            logger.warning("Gemini API key missing. Falling back to Mock LLM Provider.")
            return MockLLMProvider().generate(prompt, system_prompt)

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            payload = {"contents": [{"parts": [{"text": full_prompt}]}]}

            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                else:
                    logger.error(f"Gemini API Error {resp.status_code}: {resp.text}")
                    return MockLLMProvider().generate(prompt, system_prompt)
        except Exception as e:
            logger.error(f"Gemini Request exception: {e}")
            return MockLLMProvider().generate(prompt, system_prompt)


class OllamaLLMProvider(BaseLLMProvider):
    """
    Local Ollama Instance Provider (http://localhost:11434).
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL

    @property
    def provider_name(self) -> str:
        return f"Ollama Local ({self.model})"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            url = f"{self.base_url}/api/generate"
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            payload = {"model": self.model, "prompt": full_prompt, "stream": False}

            with httpx.Client(timeout=45.0) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    return resp.json()["response"].strip()
                else:
                    logger.error(f"Ollama API Error {resp.status_code}: {resp.text}")
                    return MockLLMProvider().generate(prompt, system_prompt)
        except Exception as e:
            logger.error(f"Ollama Connection Exception: {e}")
            return MockLLMProvider().generate(prompt, system_prompt)


class MockLLMProvider(BaseLLMProvider):
    """
    Local Zero-Cost Mock Provider extracting facts from prompt context.
    Ensures application runs 100% locally without external API dependencies.
    """

    @property
    def provider_name(self) -> str:
        return "Local Grounded Engine (Mock LLM)"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        # Extract context passages embedded in prompt
        if "RELEVANT DOCUMENT CONTEXT:" in prompt:
            context_section = prompt.split("RELEVANT DOCUMENT CONTEXT:")[1].split("USER QUESTION:")[0].strip()
            return f"Based on the provided research documents, here is the synthesized answer:\n\n{context_section[:400]}..."
        elif "TEXT TO SUMMARIZE:" in prompt:
            text_sec = prompt.split("TEXT TO SUMMARIZE:")[1].strip()
            return f"Summary of document:\n{text_sec[:300]}..."
        else:
            return "Based on the available research literature, the uploaded documents discuss machine learning architectures and data science methodologies."


class LLMProviderFactory:
    """
    Factory creating configured LLM provider instance.
    """

    @staticmethod
    def get_provider() -> BaseLLMProvider:
        prov = settings.LLM_PROVIDER.lower()
        if prov == "openai":
            return OpenAILLMProvider()
        elif prov == "gemini":
            return GeminiLLMProvider()
        elif prov == "ollama":
            return OllamaLLMProvider()
        else:
            return MockLLMProvider()
