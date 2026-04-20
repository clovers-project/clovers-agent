import magic
import base64


def data_url(data: bytes) -> str:
    mime_type, mime_subtype = magic.from_buffer(data, mime=True).split("/")
    base64_image = base64.b64encode(data).decode()
    return f"data:{mime_type}/{mime_subtype};base64,{base64_image}；"
