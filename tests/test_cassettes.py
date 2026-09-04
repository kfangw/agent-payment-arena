"""Tests for explicit recording and default model replay."""

from pathlib import Path

import pytest

from arena.providers.cassette import CassetteProvider
from arena.providers.protocol import ModelResponse


def test_cassette_records_only_when_explicit(tmp_path: Path) -> None:
    replay = CassetteProvider("test", "model", tmp_path)
    with pytest.raises(FileNotFoundError, match="cassette is missing"):
        replay.complete("prompt")

    recorder = CassetteProvider(
        "test",
        "model",
        tmp_path,
        record=True,
        live_complete=lambda prompt: ModelResponse(prompt.upper(), 2, 1),
    )
    assert recorder.complete("prompt").text == "PROMPT"
    assert replay.complete("prompt") == ModelResponse("PROMPT", 2, 1)


def test_model_identity_changes_cassette_key(tmp_path: Path) -> None:
    first = CassetteProvider("test", "first", tmp_path)
    second = CassetteProvider("test", "second", tmp_path)
    assert first._path("same") != second._path("same")
