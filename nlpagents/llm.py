"""LLM clients.

* ``OllamaClient`` – talks to a local Ollama server (default model ``llama3.1:8b``).
* ``ScriptedClient`` – returns canned replies; used in tests.

Agents ask for JSON and parse it with ``parse_json``, which tolerates the usual
small-model habits (code fences, text before/after the object).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

import requests


class LLMClient(Protocol):
    name: str

    def chat(self, system: str, user: str, json_mode: bool = True, temperature: float = 0.2) -> str: ...


@dataclass
class OllamaClient:
    model: str = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    num_ctx: int = 8192
    timeout: int = 300
    calls: int = 0
    seconds: float = 0.0

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def chat(self, system: str, user: str, json_mode: bool = True, temperature: float = 0.2) -> str:
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"temperature": temperature, "num_ctx": self.num_ctx, "seed": 42}}
        if json_mode:
            body["format"] = "json"
        t0 = time.perf_counter()
        r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        r.raise_for_status()
        self.calls += 1
        self.seconds += time.perf_counter() - t0
        return r.json()["message"]["content"]

    def available(self) -> bool:
        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=3).json()
            return any(m["name"].startswith(self.model.split(":")[0]) for m in tags.get("models", []))
        except (requests.RequestException, ValueError):
            return False


@dataclass
class ScriptedClient:
    """Replies with the next scripted answer whose key appears in the system prompt."""
    replies: dict[str, list[str]]
    name: str = "scripted"
    log: list[tuple[str, str]] = field(default_factory=list)

    def chat(self, system: str, user: str, json_mode: bool = True, temperature: float = 0.2) -> str:
        for key, queue in self.replies.items():
            if key in system and queue:
                self.log.append((key, user))
                return queue.pop(0) if len(queue) > 1 else queue[0]
        raise KeyError(f"No scripted reply for prompt starting: {system[:60]!r}")


def parse_json(text: str) -> dict:
    """Extract the first JSON object from a model reply."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)) if start >= 0 else []:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break
    raise ValueError(f"Model did not return valid JSON: {text[:200]!r}")
