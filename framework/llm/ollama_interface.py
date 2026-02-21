from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import json
import os
from urllib.error import URLError
from urllib.request import urlopen


@dataclass(frozen=True)
class OllamaInterface:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = "llama3.2"
    keep_alive: str = "5m"

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        with urlopen(f"{self.base_url}{endpoint}", data=data, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, endpoint: str) -> dict[str, Any]:
        with urlopen(f"{self.base_url}{endpoint}", timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def chat(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return self._post("/api/chat", {"model": self.model, "messages": messages, "stream": False, "keep_alive": self.keep_alive})

    def generate(self, prompt: str) -> dict[str, Any]:
        return self._post("/api/generate", {"model": self.model, "prompt": prompt, "stream": False, "keep_alive": self.keep_alive})

    def embeddings(self, text: str) -> dict[str, Any]:
        return self._post("/api/embeddings", {"model": self.model, "prompt": text, "keep_alive": self.keep_alive})

    def list_models(self) -> dict[str, Any]:
        return self._get("/api/tags")

    def keep_alive_ping(self) -> dict[str, Any]:
        return self.generate("ping")

    def is_available(self) -> bool:
        try:
            self.list_models()
            return True
        except URLError:
            return False
        except Exception:
            return False
