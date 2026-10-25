import re
import json
import base64
import contextlib
from pathlib import Path

import httpx
import chardet
import puremagic
from tenacity import RetryCallState, retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from collections.abc import Callable
from clovers.base import Coro

type AsyncFunction[**P, T] = Callable[P, Coro[T]]

B64_PATTERN = re.compile(r"^[A-Za-z0-9+/]*={0,2}$")


def is_base64(s: str) -> bool:
    return bool(len(s) % 4 == 0 and B64_PATTERN.match(s))


def data_url(data: bytes, allowed_subtype: tuple[str, ...]) -> str | None:
    mime = puremagic.from_string(data, mime=True).split("/", 2)
    if len(mime) != 2:
        return
    mime, subtype = mime
    if allowed_subtype and subtype not in allowed_subtype:
        return
    b64 = base64.b64encode(data).decode()
    return f"data:{mime}/{subtype};base64,{b64}"


def parse_b64url(s: str):
    """
    解析 data URI（仅 base64），返回 (type, subtype, 精确字节数)；
    解析失败或不支持的格式返回 None。
    """
    # 1) 快速前缀检查（大小写不敏感）
    if s[:5].lower() != "data:":
        return None
    # 2) 定位分隔逗号（header 一般很短，find 很快）
    comma = s.find(",", 5)
    if comma < 0:
        return None
    # 3) 解析 header： type/subtype;param1;param2...
    header = s[5:comma]  # header 很短，切片开销可忽略
    parts = header.split(";")
    mediatype = parts[0]
    is_b64 = any(p.strip().lower() == "base64" for p in parts[1:])
    if not is_b64:
        return None  # 非 base64 不支持，直接失败
    # 4) 拆 type / subtype
    slash = mediatype.find("/")
    if slash < 0:
        if mediatype:  # 有内容但没有 '/' → 非法
            return None
        tp, st = "text", "plain"  # data:, 默认值
    else:
        tp = mediatype[:slash]
        st = mediatype[slash + 1 :]
    # 5) 计算 base64 解码后的精确字节数（不切片、不解码）
    L = len(s) - comma - 1  # data 部分的字符数
    if L == 0:
        return tp, st, 0
    rem = L & 3
    if rem:  # 无填充
        if rem == 1:  # base64 长度非法
            return None
        n = (L * 3) >> 2  # floor(L * 3 / 4)
    else:  # 4 的倍数，看尾部 '=' 填充
        if s[-1] == "=":
            n = ((L >> 2) * 3) - (2 if s[-2] == "=" else 1)
        else:
            n = (L >> 2) * 3
    return tp, st, n


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
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.RequestError)


def retry_httpx(retries: int = 5, max_delay: float = 8.0):
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

    return retry(stop=stop_after_attempt(retries), wait=wait, retry=retry_if_exception(is_retryable_httpx_exc), reraise=True)


async def httpx_request(method: AsyncFunction[[str], httpx.Response], url: str, **kwargs):
    resp = await method(url, **kwargs)
    resp.raise_for_status()
    return resp
