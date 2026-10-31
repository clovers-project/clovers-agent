import os
import re
import json
import base64
import contextlib
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
import chardet
import puremagic
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from collections.abc import Callable, Coroutine
from typing import Any

type AsyncFunction[**P, T] = Callable[P, Coroutine[Any, Any, T]]


B64_PATTERN = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")


def is_base64(s: str) -> bool:
    return bool(len(s) % 4 == 0 and B64_PATTERN.match(s))


def to_b64url(data: bytes, allowed_subtype: tuple[str, ...]) -> str | None:
    mime = puremagic.from_string(data, mime=True).split("/", 2)
    if len(mime) != 2:
        return
    mime, subtype = mime
    if allowed_subtype and subtype not in allowed_subtype:
        return
    b64 = base64.b64encode(data).decode()
    return f"data:{mime}/{subtype};base64,{b64}"


_B64_DELETE = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_CHUNK = 1 << 16  # 64 KiB: bounds temporary allocations for huge payloads


def parse_b64uri(uri: str) -> tuple[str, str, int] | None:
    """Parse ``data:<type>/<subtype>[;param...];base64,<payload>``.

    Returns ``(type, subtype, decoded_byte_length)``, or ``None`` if *uri* is
    not a well-formed base64 data URI.

    Behaviour
    ---------
    * The ``data:`` scheme is matched case-insensitively, as required by
      RFC 3986.
    * ``;base64`` is matched case-insensitively and must appear exactly once,
      as the last header parameter, as required by RFC 2397.  Other
      parameters are ignored (but not otherwise validated).
    * An omitted media type defaults to ``text/plain``; type and subtype are
      returned lower-cased.
    * Only the standard base64 alphabet is accepted.  Whitespace and the
      URL-safe alphabet (``-``/``_``) are rejected.
    * Padding is optional (``AAA`` and ``AAA=`` are both fine), but if
      present it must be correct: one or two ``=``, only at the very end.
    """
    if not isinstance(uri, str):
        return None
    if not uri.startswith("data:") and uri[:5].lower() != "data:":
        return None

    comma = uri.find(",", 5)
    if comma < 0:
        return None

    # ---- header: ... ;base64 , --------------------------------------
    header_len = comma - 5
    if header_len < 7 or uri[comma - 7] != ";" or uri[comma - 6 : comma].lower() != "base64":
        return None

    type_end = comma - 7
    param = uri.find(";", 5, type_end)

    # RFC 2397 allows the ;base64 marker only once, as the final parameter.
    # Reject an earlier one; it would otherwise be silently ignored.
    if param >= 0:
        p = param
        while p < type_end:
            q = uri.find(";", p + 1, type_end)
            if q < 0:
                q = type_end
            if q - p == 7 and uri[p + 1 : q].lower() == "base64":
                return None
            p = q

    mediatype = uri[5 : type_end if param < 0 else param]

    if mediatype:
        slash = mediatype.find("/")
        if (
            slash < 1
            or slash == len(mediatype) - 1
            or mediatype.find("/", slash + 1) >= 0
            or not mediatype.isascii()
            or not mediatype.isprintable()
            or " " in mediatype
        ):
            return None
        mime_type = mediatype[:slash].lower()
        mime_subtype = mediatype[slash + 1 :].lower()
    else:
        mime_type, mime_subtype = "text", "plain"

    # ---- payload: validate in chunks, compute length ----------------
    start = comma + 1
    end = len(uri)
    n = end - start
    if n % 4 == 1:  # a single leftover base64 character is impossible
        return None

    # Peel off the (at most two) trailing '=' characters.
    k = end
    pad = 0
    while k > start and uri[k - 1] == "=":
        k -= 1
        pad += 1
        if pad > 2:
            return None

    if pad and n % 4:  # padding only makes sense on whole 4-char groups
        return None

    # Anything left in the payload must be a base64 alphabet character.
    try:
        for i in range(start, k, _CHUNK):
            stop = i + _CHUNK
            if stop > k:
                stop = k
            if uri[i:stop].encode("ascii").translate(None, _B64_DELETE):
                return None
    except UnicodeEncodeError:  # non-ASCII character in the payload
        return None

    # 4 base64 chars -> 3 bytes; each '=' removes one byte.
    return mime_type, mime_subtype, (n * 3 >> 2) - pad


def read_file_url(url: str) -> bytes | None:
    parsed = urlparse(url)
    if parsed.netloc not in ("", "localhost"):
        return
    path = unquote(parsed.path)
    if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
        path = path[1:]
    file = Path(path)
    if not file.is_file():
        return
    return file.read_bytes()


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


def read_file(file: Path):
    if file.exists():
        return file.read_text(encoding="utf-8")
    else:
        return None


def load_json(file: Path):
    if not file.exists():
        return
    with file.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return


def save_json(file: Path, data):
    parent = file.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
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
    parent = file.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    s = json.dumps(data, ensure_ascii=False)
    with file.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


def decode_bytes(btext: bytes | bytearray):
    encoding = chardet.detect(btext)["encoding"]
    try:
        return btext.decode(encoding=encoding or "ascii")
    except (UnicodeDecodeError, LookupError):
        return ""


def is_retryable_httpx_exc(exc: BaseException) -> bool:
    return isinstance(exc, (httpx.RequestError, httpx.HTTPStatusError))


async def httpx_request(method: AsyncFunction[..., httpx.Response], url: str, **kwargs):
    response = await method(url, **kwargs)
    status_code = response.status_code
    if status_code == 429 or status_code >= 500:
        response.raise_for_status()
    return response


def retry_httpx(max_attempts: int = 5, max_delay: float = 8.0):
    backoff = wait_random_exponential(multiplier=1, max=max_delay)

    def wait(state: RetryCallState) -> float:
        outcome = state.outcome
        if outcome is None:
            return backoff(state)
        exc = outcome.exception()
        if isinstance(exc, httpx.HTTPStatusError) and (x := exc.response.headers.get("retry-after")):
            with contextlib.suppress(ValueError):
                return min(float(x), max_delay)
        return backoff(state)

    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait,
        retry=retry_if_exception(is_retryable_httpx_exc),
        reraise=True,
    )(httpx_request)
