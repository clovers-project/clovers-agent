import hashlib
import os
import stat
from collections import deque
from datetime import datetime
from pathlib import Path

from typing import NamedTuple


def format_path(path: str):
    return f".{path[10:]}" if path.startswith("/workspace") else path.lstrip("/\\")


def format_size(size: float) -> str:
    if size < 1024:
        return f"{size:.0f}B"
    for unit in ("K", "M", "G"):
        size /= 1024
        if size < 1024:
            break
    return f"{size:.2f}{unit}"


def md5(data: bytes):
    hash_md5 = hashlib.md5()
    hash_md5.update(data)
    return hash_md5.hexdigest()


MAX_LS_ENTRIES = 200


class LsEntry(NamedTuple):
    name: str
    kind: str
    size: int | None
    mtime: str | None


def _is_junction(entry: os.DirEntry) -> bool:
    # Windows junctions lstat as directories but can point at an ancestor, so never recurse into one.
    is_junction = getattr(entry, "is_junction", None)
    if is_junction is None:
        return False
    try:
        return is_junction()
    except OSError:
        return False


def _format_mtime(timestamp: float) -> str:
    try:
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        return "?"


def _first_visit(visited: set[tuple[int, int]], st: os.stat_result) -> bool:
    if not st.st_ino:
        return True
    ident = (st.st_dev, st.st_ino)
    if ident in visited:
        return False
    visited.add(ident)
    return True


def list_entries(root: Path, depth: int) -> list[LsEntry]:
    root_stat = root.stat()
    if not stat.S_ISDIR(root_stat.st_mode):
        raise NotADirectoryError(str(root))

    entries: list[LsEntry] = []
    queue: deque[tuple[Path, int]] = deque([(root, 0)])
    visited: set[tuple[int, int]] = set()
    _first_visit(visited, root_stat)

    while queue:
        current, level = queue.popleft()
        if level >= depth:
            continue
        try:
            with os.scandir(current) as scanner:
                children = list(scanner)
        except OSError:
            if level == 0:
                raise
            continue  # vanished or unreadable since it was queued: skip it, keep the rest
        for child in children:
            name = Path(child.path).relative_to(root).as_posix()
            try:
                st = child.stat(follow_symlinks=False)
            except OSError:
                entries.append(LsEntry(name, "?", None, None))
                continue
            is_link = stat.S_ISLNK(st.st_mode) or (stat.S_ISDIR(st.st_mode) and _is_junction(child))
            is_dir = stat.S_ISDIR(st.st_mode) and not is_link
            entries.append(
                LsEntry(
                    name,
                    "d" if is_dir else "l" if is_link else "f",
                    None if is_dir or is_link else st.st_size,
                    _format_mtime(st.st_mtime),
                )
            )
            if is_dir and _first_visit(visited, st):
                queue.append((Path(child.path), level + 1))
    return entries


def render_listing(path: str, depth: int, entries: list[LsEntry]) -> str:
    entries = sorted(entries, key=lambda entry: (entry.kind != "d", entry.name))
    lines = [
        f"Contents of `{path}` (depth {depth}):\n",
        "kind|size|mtime|name",
        "---|---|---|---",
    ]
    for entry in entries[:MAX_LS_ENTRIES]:
        if entry.kind == "f" and entry.size is not None:
            size = format_size(entry.size)
        else:
            size = "?" if entry.kind == "?" else "-"
        lines.append(f"{entry.kind}|{size}|{entry.mtime or '?'}|{entry.name}")
    if len(entries) > MAX_LS_ENTRIES:
        lines.append(f"\n(truncated at {MAX_LS_ENTRIES} entries)")
    return "\n".join(lines)
