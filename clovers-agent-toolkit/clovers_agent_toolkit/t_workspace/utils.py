import hashlib


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
