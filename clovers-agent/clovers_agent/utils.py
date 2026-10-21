import re
import json
import base64
from pathlib import Path

import chardet
import puremagic

B64_PATTERN = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")


def is_base64(s: str) -> bool:
    return bool(len(s) % 4 == 0 and B64_PATTERN.match(s))


def data_url(data: bytes, allowed_subtype: tuple[str, ...]) -> str | None:
    mime = puremagic.from_string(data, mime=True).split("/", 2)
    if len(mime) != 2:
        return
    subtype = mime[1]
    if allowed_subtype and subtype not in allowed_subtype:
        return
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def deep_add(total: dict, detail: dict):
    """递归累加两个字典中的数值（仅限整数）"""
    if not isinstance(detail, dict):
        return total
    for k, v in detail.items():
        if isinstance(v, dict):
            if k not in total:
                total[k] = {}
            total[k] = deep_add(total[k], v)
        elif isinstance(v, int):
            if k not in total:
                total[k] = 0
            elif not isinstance(total[k], int):
                del total[k]
                continue
            total[k] = total[k] + v
    return total


def read_file(file: Path, default_text: str):
    if file.exists():
        return file.read_text(encoding="utf-8")
    else:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(default_text, encoding="utf-8")
        return default_text


def load_json(file: Path):
    if not file.exists():
        return
    with file.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return


def save_json(file: Path, data):
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.parent / (file.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    tmp.replace(file)


def load_jsonl(file: Path):
    if not file.exists():
        return
    arr = []
    with file.open("r", encoding="utf-8") as f:
        try:
            arr.extend(json.loads(data) for line in f if (data := line.strip()))
        except json.JSONDecodeError:
            return
    return arr


def save_jsonl(file: Path, data):
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def decode_bytes(btext: bytes | bytearray):
    encoding = chardet.detect(btext)["encoding"]
    try:
        return btext.decode(encoding=encoding or "ascii")
    except (UnicodeDecodeError, LookupError):
        return ""
