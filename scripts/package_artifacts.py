"""Create and verify a deterministic archive of result artifacts."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path


def digest(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def collect(specs: list[str]) -> list[tuple[str, Path]]:
    """Resolve labeled inputs into stable archive paths."""
    files: list[tuple[str, Path]] = []
    for spec in specs:
        label, separator, raw_path = spec.partition("=")
        if not separator or not label or "/" in label:
            raise ValueError(f"input must be LABEL=PATH: {spec}")
        source = Path(raw_path).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file():
            files.append((f"{label}/{source.name}", source))
        else:
            files.extend(
                (f"{label}/{path.relative_to(source).as_posix()}", path)
                for path in sorted(source.rglob("*"))
                if path.is_file()
            )
    names = [name for name, _ in files]
    if len(names) != len(set(names)):
        raise ValueError("archive paths collide")
    return sorted(files)


def create_archive(inputs: list[str], output: Path) -> None:
    """Write a normalized archive and its external metadata record."""
    files = collect(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"path": name, "bytes": source.stat().st_size, "sha256": digest(source)}
        for name, source in files
    ]
    manifest = json.dumps({"schema": 1, "files": rows}, indent=2, sort_keys=True).encode() + b"\n"
    with (
        output.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped,
        tarfile.open(fileobj=zipped, mode="w") as archive,
    ):
        for name, source in files:
            info = archive.gettarinfo(str(source), arcname=name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with source.open("rb") as stream:
                archive.addfile(info, stream)
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(manifest)
        info.mode = 0o644
        info.mtime = 0
        archive.addfile(info, __import__("io").BytesIO(manifest))
    metadata = {
        "schema": 1,
        "created_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "archive": output.name,
        "archive_sha256": digest(output),
        "file_count": len(files),
        "uncompressed_bytes": sum(row["bytes"] for row in rows),
        "input_labels": [spec.partition("=")[0] for spec in inputs],
    }
    output.with_suffix(output.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def verify_archive(path: Path) -> None:
    """Verify every archived file against the embedded manifest."""
    with tarfile.open(path, "r:gz") as archive:
        manifest_member = archive.getmember("MANIFEST.json")
        manifest_file = archive.extractfile(manifest_member)
        if manifest_file is None:
            raise RuntimeError("cannot read MANIFEST.json")
        manifest = json.load(manifest_file)
        expected = {row["path"]: row for row in manifest["files"]}
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        actual_names = set(members) - {"MANIFEST.json"}
        if actual_names != set(expected):
            raise RuntimeError("archive contents do not match the manifest")
        for name, row in expected.items():
            stream = archive.extractfile(members[name])
            if stream is None:
                raise RuntimeError(f"cannot read {name}")
            value = hashlib.sha256()
            for chunk in iter(lambda source=stream: source.read(1024 * 1024), b""):
                value.update(chunk)
            if value.hexdigest() != row["sha256"] or members[name].size != row["bytes"]:
                raise RuntimeError(f"checksum failure: {name}")
    print(f"verified {len(expected)} files: {path}")


def main() -> None:
    """Run the archive command-line interface."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("pack")
    pack.add_argument("--input", action="append", required=True, help="LABEL=PATH")
    pack.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.command == "pack":
        create_archive(args.input, args.output)
        verify_archive(args.output)
    else:
        verify_archive(args.archive)


if __name__ == "__main__":
    main()
