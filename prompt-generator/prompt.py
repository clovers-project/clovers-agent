import re
import tomllib
import requests
from pathlib import Path

LOCAL_PATH = Path(__file__).parent

TASK = """\
帮我构建一个工具调用ai的提示词，
这个 ai 承担的角色是语义分发
会拿到上下文和用户对话，判断这个对话属于哪个工具定义的场景

下面是一些场景的描述

当前对话为闲聊、或不属于其他工具，调用此方法。
当用户发出指令、或需要特定功能支持时，调用此方法以进入技能执行环境。
当助手被群友进行色情调戏时必须调用此方法以获取回复准则。
...
注意语义场景是可扩展的
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
