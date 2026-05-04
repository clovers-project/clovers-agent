import base64
import puremagic


def is_base64(s: str) -> bool:
    import re

    return bool(len(s) % 4 == 0 and re.compile(r"^[A-Za-z0-9+/]*={0,2}$").match(s))


def data_url(data: bytes) -> str:
    mime = puremagic.from_string(data, mime=True)
    base64_image = base64.b64encode(data).decode()
    return f"data:{mime};base64,{base64_image}"


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
