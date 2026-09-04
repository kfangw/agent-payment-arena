"""Explicit-record, replay-by-default model response cassettes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from arena.providers.protocol import ModelResponse


class CassetteProvider:
    """Replay recorded responses, calling a provider only in record mode."""

    def __init__(
        self,
        provider_id: str,
        model_id: str,
        directory: Path,
        *,
        record: bool = False,
        live_complete: Callable[[str], ModelResponse] | None = None,
    ) -> None:
        self.provider_id = provider_id
        self.model_id = model_id
        self.directory = directory
        self.record = record
        self.live_complete = live_complete

    def complete(self, prompt: str) -> ModelResponse:
        """Replay a response or deliberately record a live call."""
        path = self._path(prompt)
        if path.exists() and not self.record:
            data = json.loads(path.read_text())
            return ModelResponse(
                text=str(data["text"]),
                prompt_tokens=int(data["prompt_tokens"]),
                completion_tokens=int(data["completion_tokens"]),
            )
        if not self.record:
            raise FileNotFoundError(f"cassette is missing: {path}")
        if self.live_complete is None:
            raise RuntimeError("record mode requires a live provider callback")
        response = self.live_complete(prompt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(response.__dict__, indent=2, sort_keys=True) + "\n")
        return response

    def _path(self, prompt: str) -> Path:
        key = hashlib.sha256(f"{self.provider_id}\0{self.model_id}\0{prompt}".encode()).hexdigest()
        return self.directory / self.provider_id / self.model_id / f"{key}.json"
