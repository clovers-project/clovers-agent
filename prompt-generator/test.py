import json
import tomllib
import requests
from pathlib import Path

LOCAL_PATH = Path(__file__).parent


payload = {
    "deepseek-v4-flash": {
        "prompt_tokens": 1174,
        "completion_tokens": 92,
        "total_tokens": 1266,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": 70},
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 1174,
    }
}


def main():

    CONFIG_PATH = LOCAL_PATH / "config.toml"
    if not CONFIG_PATH.exists():
        print(f"配置文件不存在，请于 {CONFIG_PATH.resolve().as_posix()} 填写正确的配置文件。")
        CONFIG_PATH.write_text('url = ""\nmodel = ""\napi_key = ""')
        return
    with (LOCAL_PATH / "config.toml").open("rb") as f:
        CONFIG: dict[str, str] = tomllib.load(f)
    message = requests.post(
        CONFIG["url"].lstrip("/") + "/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {CONFIG['api_key']}", "Content-Type": "application/json"},
    ).json()
    print(json.dumps(message, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
