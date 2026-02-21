from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import json
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass(frozen=True)
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    keep_alive: str = "5m"

    def generate(self, prompt: str) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False, "keep_alive": self.keep_alive}
        data = json.dumps(payload).encode("utf-8")
        req_url = f"{self.base_url}/api/generate"
        with urlopen(req_url, data=data, timeout=10) as resp:
            body: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return str(body.get("response", ""))

    def embeddings(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text, "keep_alive": self.keep_alive}
        data = json.dumps(payload).encode("utf-8")
        req_url = f"{self.base_url}/api/embeddings"
        with urlopen(req_url, data=data, timeout=10) as resp:
            body: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        embedding = body.get("embedding", [])
        return [float(x) for x in embedding]


@dataclass(frozen=True)
class DSPyLikeRepl:
    client: OllamaClient

    def run_turn(self, prompt: str) -> dict[str, Any]:
        completion = self.client.generate(prompt)
        vectors = self.client.embeddings(completion[:256] if completion else "")
        return {
            "prompt": prompt,
            "completion": completion,
            "embedding_size": len(vectors),
            "model": self.client.model,
            "keep_alive": self.client.keep_alive,
        }
