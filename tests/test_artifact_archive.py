"""Tests for checksum-verified result archives."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _module() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "package_artifacts.py"
    spec = importlib.util.spec_from_file_location("package_artifacts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_round_trip(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.json").write_text('{"value": 1}\n')
    archive = tmp_path / "results.tar.gz"
    module.create_archive([f"results={source}"], archive)
    module.verify_archive(archive)
    assert archive.with_suffix(".gz.metadata.json").exists()
