import base64
import puremagic


def is_base64(s: str) -> bool:
    import re

    return bool(len(s) % 4 == 0 and re.compile(r"^[A-Za-z0-9+/]*={0,2}$").match(s))


def data_url(data: bytes) -> str:
    mime = puremagic.from_string(data, mime=True)
    base64_image = base64.b64encode(data).decode()
    return f"data:{mime};base64,{base64_image}"
