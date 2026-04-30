import base64
import puremagic


def data_url(data: bytes) -> str:
    mime = puremagic.from_string(data, mime=True)
    base64_image = base64.b64encode(data).decode()
    return f"data:{mime};base64,{base64_image}"
