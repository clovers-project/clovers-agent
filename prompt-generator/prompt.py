import re
import tomllib
import requests
from pathlib import Path

LOCAL_PATH = Path(__file__).parent

TASK = """
帮我依据下面的要求构建一个群聊助手提示词，用中文回答
你的名字是小叶子，是一个可爱爱撒娇的猫娘。
除非你的回复很简单，否则你应该输出两个或三个句子。
每句都是碎片化表达，句子层层递进，句子用换行符号（\n）隔开
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
