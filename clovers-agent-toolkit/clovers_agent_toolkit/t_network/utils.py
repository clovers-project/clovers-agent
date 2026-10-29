import httpx
from collections.abc import Awaitable


async def safe_resp[T](coro: Awaitable[T]) -> T | str:
    try:
        return await coro
    except httpx.HTTPStatusError as e:
        resp = e.response
        try:
            return f"请求失败（HTTP {resp.status_code}）：{resp.json()["error"]["message"]}"
        except Exception:
            return f"请求失败（HTTP {resp.status_code}）"
    except Exception:
        return "请求失败"
