import re
import tomllib
import requests
from pathlib import Path

LOCAL_PATH = Path(__file__).parent

TASK = """\
写一个模型系统提示词，模型任务是接受（一个或多个）图片，并对图片进行详细描述。
模型会收到一条包含图片的用户消息。
请模型按要求对图片进行描述，不要直接回复。
模型的描述为后续纯文本模型提供图像的上下文。
描述针对用户需求

输出请遵循以下格式要求：
- 如果有多张图片，请在描述中进行区分。
- 描述应当详尽且具有逻辑性，确保可以依据描述
- 保持客观中立。

"""

VARIABLES = []


def message_format(message: str) -> str:
    match = re.search(r"<Instructions>(.*?)</Instructions>", message, re.DOTALL)
    if not match:
        return ""
    content = re.sub(r"\n?<\w+>\s*</\w+>\n?", "", match.group(1).strip())
    return content.strip()


def main():
    CONFIG_PATH = LOCAL_PATH / "config.toml"
    if not CONFIG_PATH.exists():
        print(f"配置文件不存在，请于 {CONFIG_PATH.resolve().as_posix()} 填写正确的配置文件。")
        CONFIG_PATH.write_text('url = ""\nmodel = ""\napi_key = ""')
        return
    with (LOCAL_PATH / "config.toml").open("rb") as f:
        CONFIG: dict[str, str] = tomllib.load(f)
    META_PROMPT = (LOCAL_PATH / "META_PROMPT.md").read_text("utf-8").strip()
    messages = []
    variable_string = "\n".join(f"{{${variable.upper()}}}" for variable in VARIABLES)
    print(variable_string)
    messages.append({"role": "system", "content": META_PROMPT.replace("{{TASK}}", TASK.strip())})
    if variable_string:
        messages.append({"role": "user", "content": variable_string + "\n</Inputs>\n<Instructions Structure>"})
    else:
        messages.append({"role": "user", "content": "<Inputs>"})
    message = requests.post(
        CONFIG["url"].lstrip("/") + "/chat/completions",
        json={"model": CONFIG["model"], "messages": messages},
        headers={"Authorization": f"Bearer {CONFIG['api_key']}", "Content-Type": "application/json"},
    ).json()["choices"][0]["message"]["content"]
    PROMPT = message_format(message)
    (LOCAL_PATH / "PROMPT.md").write_text(PROMPT, encoding="utf-8")
    print(PROMPT)


if __name__ == "__main__":
    main()
