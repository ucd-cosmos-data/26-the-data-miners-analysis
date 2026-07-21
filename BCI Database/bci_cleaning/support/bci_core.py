"""Shared, safety-focused utilities for the Dreyer EEG cleaning pipeline."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import mimetypes
import os
import re
import shutil
import ssl
import stat
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = PROJECT_ROOT / "raw"
DEFAULT_CLEANED = PROJECT_ROOT / "cleaned"
DEFAULT_REPORTS = PROJECT_ROOT / "reports"
DEFAULT_LOGS = PROJECT_ROOT / "logs"
ZENODO_URL = "https://zenodo.org/records/8089820/files/BCI%20Database.zip?download=1"
ZENODO_RECORD = "https://zenodo.org/records/8089820"
ARTICLE_URL = "https://www.nature.com/articles/s41597-023-02445-z"
ARCHIVE_MD5 = "c8c6f5f09f1882666eec10e29155fdcb"
CHUNK_SIZE = 8 * 1024 * 1024

ISSUE_FIELDS = [
    "issue_id",
    "severity",
    "phase",
    "relative_path",
    "participant_run",
    "check",
    "observed_value",
    "expected_rule",
    "documentation_source",
    "resolution",
    "status",
]

INVENTORY_FIELDS = [
    "relative_path",
    "basename",
    "extension",
    "size_bytes",
    "mtime_ns",
    "mode",
    "readable",
    "mime_guess",
    "sha256",
    "crc32",
    "duplicate_basename",
    "duplicate_content",
]

MANIFEST_FIELDS = [
    "raw_relative_path",
    "cleaned_relative_path",
    "raw_sha256",
    "cleaned_sha256",
    "action",
    "reason",
    "documentation_source",
    "equivalence_check",
    "status",
]


@dataclass(frozen=True)
class Issue:
    severity: str
    phase: str
    relative_path: str
    participant_run: str
    check: str
    observed_value: str
    expected_rule: str
    documentation_source: str
    resolution: str
    status: str = "open"
    issue_id: str = ""

    def row(self) -> dict[str, str]:
        payload = asdict(self)
        if not payload["issue_id"]:
            stable = "|".join(
                str(payload[key])
                for key in ("phase", "relative_path", "check", "observed_value")
            )
            payload["issue_id"] = hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]
        return {key: str(payload.get(key, "")) for key in ISSUE_FIELDS}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return str(uuid.uuid4())


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    ensure_directory(path.parent)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding=encoding, newline="", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    ensure_directory(path.parent)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    ensure_directory(path.parent)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def log_event(logs: Path, run_id: str, phase: str, event: str, **details: Any) -> None:
    append_jsonl(
        logs / "pipeline.jsonl",
        {
            "timestamp": utc_now(),
            "run_id": run_id,
            "phase": phase,
            "event": event,
            **details,
        },
    )


def merge_issues(path: Path, issues: Iterable[Issue], phases: set[str]) -> None:
    retained = [row for row in read_csv_dicts(path) if row.get("phase") not in phases]
    fresh = [issue.row() for issue in issues]
    combined = sorted(
        retained + fresh,
        key=lambda row: (
            row.get("severity", ""),
            row.get("phase", ""),
            row.get("relative_path", ""),
            row.get("check", ""),
        ),
    )
    atomic_write_csv(path, combined, ISSUE_FIELDS)


def has_blocking_issues(path: Path) -> bool:
    return any(row.get("severity") in {"fatal", "error"} for row in read_csv_dicts(path))


def common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--logs", type=Path, default=DEFAULT_LOGS)
    return parser


def resolve_raw(raw: Path) -> Path:
    if not raw.exists():
        raise FileNotFoundError(f"Raw dataset does not exist: {raw}")
    resolved = raw.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"Raw dataset is not a directory: {resolved}")
    return resolved


def assert_safe_paths(raw: Path, cleaned: Path) -> tuple[Path, Path]:
    raw_resolved = resolve_raw(raw)
    cleaned_abs = cleaned.resolve(strict=False)
    try:
        cleaned_abs.relative_to(raw_resolved)
    except ValueError:
        pass
    else:
        embedded_cleaned = (PROJECT_ROOT / "cleaned").resolve(strict=False)
        if cleaned_abs != embedded_cleaned:
            raise RuntimeError("Refusing to place cleaned output inside the raw dataset")
    try:
        raw_resolved.relative_to(cleaned_abs)
    except ValueError:
        pass
    else:
        raise RuntimeError("Refusing to place raw data inside the cleaned output")
    return raw_resolved, cleaned_abs


def iter_files(root: Path) -> Iterator[Path]:
    root = root.resolve(strict=True)
    project = PROJECT_ROOT.resolve(strict=True)
    try:
        project.relative_to(root)
    except ValueError:
        excluded_project: Path | None = None
    else:
        excluded_project = project

    files: list[Path] = []
    for current_text, directories, filenames in os.walk(root, followlinks=False):
        current = Path(current_text)
        if excluded_project is not None:
            directories[:] = [
                name
                for name in directories
                if (current / name).resolve(strict=False) != excluded_project
            ]
        directories.sort()
        for name in sorted(filenames):
            path = current / name
            if path.is_file():
                files.append(path)
    yield from sorted(files, key=lambda path: path.relative_to(root).as_posix())


def extension_for(path: Path) -> str:
    if path.name.startswith(".") and path.name.count(".") == 1:
        return ""
    return path.suffix.lower().lstrip(".")


def file_hashes(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    crc = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
    return digest.hexdigest(), f"{crc & 0xFFFFFFFF:08x}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def load_hash_cache(path: Path) -> dict[tuple[str, int, int], tuple[str, str]]:
    cache: dict[tuple[str, int, int], tuple[str, str]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                key = (row["relative_path"], int(row["size_bytes"]), int(row["mtime_ns"]))
                cache[key] = (row["sha256"], row["crc32"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return cache


def inventory_rows(raw: Path, logs: Path, run_id: str) -> list[dict[str, Any]]:
    cache_path = logs / "inventory_hash_cache.jsonl"
    cache = load_hash_cache(cache_path)
    rows: list[dict[str, Any]] = []
    paths = list(iter_files(raw))
    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(raw).as_posix()
        info = path.stat()
        key = (relative, info.st_size, info.st_mtime_ns)
        cached = cache.get(key)
        if cached is None:
            sha256, crc32 = file_hashes(path)
            append_jsonl(
                cache_path,
                {
                    "relative_path": relative,
                    "size_bytes": info.st_size,
                    "mtime_ns": info.st_mtime_ns,
                    "sha256": sha256,
                    "crc32": crc32,
                },
            )
        else:
            sha256, crc32 = cached
        rows.append(
            {
                "relative_path": relative,
                "basename": path.name,
                "extension": extension_for(path),
                "size_bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "mode": stat.filemode(info.st_mode),
                "readable": str(os.access(path, os.R_OK)).lower(),
                "mime_guess": mimetypes.guess_type(path.name)[0] or "",
                "sha256": sha256,
                "crc32": crc32,
            }
        )
        if index % 25 == 0 or index == len(paths):
            log_event(logs, run_id, "inventory", "hash_progress", completed=index, total=len(paths))
    basename_counts = Counter(row["basename"] for row in rows)
    hash_counts = Counter(row["sha256"] for row in rows)
    for row in rows:
        row["duplicate_basename"] = str(basename_counts[row["basename"]] > 1).lower()
        row["duplicate_content"] = str(hash_counts[row["sha256"]] > 1).lower()
    return rows


def write_structure_report(raw: Path, inventory: Sequence[dict[str, Any]], destination: Path) -> None:
    extension_counts = Counter(row["extension"] or "[no extension]" for row in inventory)
    file_paths = [Path(row["relative_path"]) for row in inventory]
    directory_paths: set[Path] = set()
    for file_path in file_paths:
        directory_paths.update(parent for parent in file_path.parents if parent != Path("."))
    dirs = sorted(directory_paths, key=lambda path: path.as_posix())
    lines = [
        "# Dataset Structure",
        "",
        f"- Raw root: `{raw}`",
        f"- Files: {len(inventory):,}",
        f"- Directories: {len(dirs) + 1:,}",
        f"- Size: {sum(int(row['size_bytes']) for row in inventory):,} bytes",
        "",
        "## File counts by type",
        "",
        "| Extension | Count |",
        "|---|---:|",
    ]
    for extension, count in sorted(extension_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{extension}` | {count:,} |")
    lines.extend(["", "## Complete recursive tree", "", "```text", f"{raw.name}/"])
    all_paths = sorted([*dirs, *file_paths], key=lambda path: path.as_posix())
    for relative in all_paths:
        depth = len(relative.parts) - 1
        suffix = "/" if relative in directory_paths else ""
        lines.append(f"{'    ' * depth}{relative.name}{suffix}")
    lines.extend(["```", ""])
    atomic_write_text(destination, "\n".join(lines))


class HTTPRangeReader(io.RawIOBase):
    """Minimal seekable HTTP reader that refuses non-range responses."""

    def __init__(self, url: str, timeout: int = 60):
        self.url = url
        self.timeout = timeout
        self.pos = 0
        try:
            import certifi
        except ImportError:
            self.ssl_context = ssl.create_default_context()
        else:
            self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.length = self._discover_length()

    def _request(self, start: int, end: int) -> bytes:
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "COSMOS-BCI-validator/1.0"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context) as response:
            if getattr(response, "status", None) != 206:
                raise RuntimeError("Zenodo did not honor the byte-range request; refusing a full download")
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-"):
                raise RuntimeError(f"Unexpected Content-Range: {content_range}")
            return response.read()

    def _discover_length(self) -> int:
        request = urllib.request.Request(
            self.url,
            headers={"Range": "bytes=0-0", "User-Agent": "COSMOS-BCI-validator/1.0"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context) as response:
            if getattr(response, "status", None) != 206:
                raise RuntimeError("Zenodo did not honor the size probe; refusing a full download")
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes 0-0/(\d+)", content_range)
            if not match:
                raise RuntimeError(f"Unexpected size response: {content_range}")
            response.read(1)
            return int(match.group(1))

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self.pos + offset
        elif whence == io.SEEK_END:
            target = self.length + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if target < 0:
            raise ValueError("Negative seek position")
        self.pos = min(target, self.length)
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if self.pos >= self.length:
            return b""
        if size is None or size < 0:
            size = self.length - self.pos
        if size > 128 * 1024 * 1024:
            raise RuntimeError("Refusing an unexpectedly large range read")
        end = min(self.pos + size, self.length) - 1
        payload = self._request(self.pos, end)
        self.pos += len(payload)
        return payload


def fetch_archive_manifest(url: str = ZENODO_URL) -> tuple[int, list[dict[str, Any]]]:
    reader = HTTPRangeReader(url)
    with zipfile.ZipFile(reader) as archive:
        rows = [
            {
                "archive_path": info.filename,
                "size_bytes": info.file_size,
                "compressed_size_bytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
            }
            for info in archive.infolist()
            if not info.is_dir()
        ]
    return reader.length, rows


def strip_archive_prefix(paths: Sequence[str], raw_name: str) -> dict[str, str]:
    prefix = raw_name.rstrip("/") + "/"
    if paths and all(path.startswith(prefix) for path in paths):
        return {path[len(prefix) :]: path for path in paths}
    first_parts = {path.split("/", 1)[0] for path in paths if "/" in path}
    if len(first_parts) == 1:
        root_prefix = next(iter(first_parts)) + "/"
        return {path[len(root_prefix) :]: path for path in paths}
    return {path: path for path in paths}


def compare_archive(
    raw_name: str,
    inventory: Sequence[dict[str, Any]],
    archive_rows: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    archive_lookup_by_full = {row["archive_path"]: row for row in archive_rows}
    normalized = strip_archive_prefix(list(archive_lookup_by_full), raw_name)
    local_lookup = {row["relative_path"]: row for row in inventory}
    results: list[dict[str, str]] = []
    for relative in sorted(set(local_lookup) | set(normalized)):
        local = local_lookup.get(relative)
        archive_full = normalized.get(relative)
        archive = archive_lookup_by_full.get(archive_full) if archive_full else None
        if local is None:
            status = "missing_local"
        elif archive is None:
            status = "extra_local"
        elif int(local["size_bytes"]) != int(archive["size_bytes"]):
            status = "size_mismatch"
        elif str(local["crc32"]).lower() != str(archive["crc32"]).lower():
            status = "crc_mismatch"
        else:
            status = "match"
        results.append(
            {
                "relative_path": relative,
                "archive_path": archive_full or "",
                "local_size_bytes": "" if local is None else str(local["size_bytes"]),
                "archive_size_bytes": "" if archive is None else str(archive["size_bytes"]),
                "local_crc32": "" if local is None else str(local["crc32"]),
                "archive_crc32": "" if archive is None else str(archive["crc32"]),
                "status": status,
            }
        )
    return results


def archive_comparison_passed(path: Path) -> bool:
    rows = read_csv_dicts(path)
    accepted = {"match", "administrative_extra"}
    return bool(rows) and all(row.get("status") in accepted for row in rows)


def detect_text_encoding(payload: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            payload.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("unknown", payload, 0, 1, "no supported encoding")


def sniff_delimiter(text: str) -> str:
    try:
        return csv.Sniffer().sniff(text[:16384], delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def parse_csv_text(text: str, delimiter: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))


def serialize_csv_rows(rows: Sequence[Sequence[str]], delimiter: str) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, delimiter=delimiter, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerows(rows)
    return output.getvalue()


def normalize_frequency_csv_bytes(payload: bytes) -> tuple[bytes, str]:
    if b"\r\r\n" not in payload:
        raise ValueError("Frequency CSV does not contain the documented CRCRLF artifact")
    if payload.replace(b"\r\r\n", b"").find(b"\r") != -1:
        raise ValueError("Frequency CSV contains additional CR bytes; preservation is ambiguous")
    text = payload.decode("utf-8")
    original_rows = parse_csv_text(text, ",")
    nonempty = [row for row in original_rows if any(cell != "" for cell in row)]
    if len(nonempty) * 2 != len(original_rows):
        raise ValueError("Unexpected empty-row pattern in frequency CSV")
    normalized_text = serialize_csv_rows(nonempty, ",")
    roundtrip = parse_csv_text(normalized_text, ",")
    if roundtrip != nonempty:
        raise ValueError("Frequency CSV semantic equivalence check failed")
    return normalized_text.encode("utf-8"), "nonempty-cell-matrix-equal"


def normalize_performances_csv_bytes(payload: bytes) -> tuple[bytes, str]:
    text = payload.decode("cp1252")
    original_rows = parse_csv_text(text, ";")
    normalized_text = serialize_csv_rows(original_rows, ";")
    roundtrip = parse_csv_text(normalized_text, ";")
    if roundtrip != original_rows:
        raise ValueError("Performance CSV semantic equivalence check failed")
    return normalized_text.encode("utf-8"), "parsed-cell-matrix-equal"


def is_administrative_artifact(relative: str, path: Path) -> bool:
    name = Path(relative).name
    if name == ".DS_Store":
        return True
    if name.startswith("~$") and path.suffix.lower() in {".docx", ".xlsx"}:
        return not zipfile.is_zipfile(path)
    return False


def clone_tree_apfs(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.is_dir() and not any(destination.iterdir()):
            destination.rmdir()
        else:
            raise FileExistsError(f"Cleaned destination must not exist or must be empty: {destination}")
    ensure_directory(destination.parent)
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    try:
        nested_destination = destination.relative_to(source)
    except ValueError:
        commands = [["cp", "-cR", str(source), str(destination)]]
    else:
        if not nested_destination.parts:
            raise RuntimeError("Cleaned destination cannot be the raw source")
        excluded_top_level = nested_destination.parts[0]
        destination.mkdir(parents=True)
        commands = [
            ["cp", "-cR", str(child), str(destination / child.name)]
            for child in sorted(source.iterdir(), key=lambda path: path.name)
            if child.name != excluded_top_level
        ]
    for command in commands:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"APFS clone failed; ordinary copy is forbidden: {result.stderr.strip()}"
            )


def verify_inventory_snapshot(raw: Path, inventory_path: Path, logs: Path, run_id: str) -> list[str]:
    expected = {row["relative_path"]: row for row in read_csv_dicts(inventory_path)}
    current_paths = {path.relative_to(raw).as_posix(): path for path in iter_files(raw)}
    errors: list[str] = []
    if set(expected) != set(current_paths):
        errors.append("Raw file membership differs from dataset_inventory.csv")
        return errors
    for index, relative in enumerate(sorted(expected), start=1):
        path = current_paths[relative]
        row = expected[relative]
        info = path.stat()
        if info.st_size != int(row["size_bytes"]):
            errors.append(f"Size changed: {relative}")
            continue
        current_hash = sha256_file(path)
        if current_hash != row["sha256"]:
            errors.append(f"SHA-256 changed: {relative}")
        if index % 25 == 0 or index == len(expected):
            log_event(logs, run_id, "snapshot", "verify_progress", completed=index, total=len(expected))
    return errors


def participant_id_from_path(relative: str) -> str:
    match = re.search(r"(?:^|/)([ABC]\d+)(?:/|$)", relative)
    return match.group(1) if match else ""


def run_id_from_name(name: str) -> str:
    match = re.search(r"_(R[1-6])_", name)
    if match:
        return match.group(1)
    if "_OE_" in name:
        return "OE"
    if "_CE_" in name:
        return "CE"
    return ""


def expected_participants() -> set[str]:
    return (
        {f"A{index}" for index in range(1, 61)}
        | {f"B{index}" for index in range(61, 82)}
        | {f"C{index}" for index in range(82, 88)}
    )


def validate_xml(path: Path) -> None:
    ET.parse(path)


def validate_zip_container(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise zipfile.BadZipFile(f"Corrupted member: {bad}")


def validate_python(path: Path) -> None:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def validate_pdf(path: Path) -> int:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required for PDF integrity validation") from exc
    reader = PdfReader(path, strict=True)
    if reader.is_encrypted:
        raise RuntimeError("Encrypted PDF cannot be validated")
    for page in reader.pages:
        _ = page.mediabox
    return len(reader.pages)


def validate_gdf_signature(path: Path) -> str:
    with path.open("rb") as handle:
        header = handle.read(8)
    if not header.startswith(b"GDF "):
        raise ValueError("Missing GDF signature")
    return header.decode("ascii", errors="strict").strip()


def disk_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def severity_counts(issues: Iterable[Issue | dict[str, str]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for issue in issues:
        if isinstance(issue, Issue):
            counts[issue.severity] += 1
        else:
            counts[issue.get("severity", "unknown")] += 1
    return counts
