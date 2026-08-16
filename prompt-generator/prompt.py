import re
import tomllib
import requests
from pathlib import Path

LOCAL_PATH = Path(__file__).parent

TASK = """\
写一段中文提示词，让ai针对用户提供的指数收集最新的新闻
1. 行业定义与基本面

先确认行业边界，以及主要产品、上下游和核心参与者。

行业主要产品/服务
上游原材料与供应商
下游客户与应用场景
主要上市公司/龙头企业
2. 供需与景气度

这是行业分析最核心的一层。

产能、产量、库存
需求量及增速
供需缺口/过剩
产品价格变化
开工率、订单量等高频指标
近期行业景气度是在上升还是下降

重点判断：

需求在增长，还是供给在扩张？

3. 行业竞争格局

判断行业里的企业有没有议价能力。

市场规模及增长率
龙头企业及市场份额
行业集中度
新进入者情况
价格战/产能竞争
技术迭代和替代风险
行业是否存在明显的规模、技术、渠道壁垒
4. 政策与宏观环境

尤其关注对整个行业产生系统性影响的因素。

最新政策、监管措施
产业扶持/限制政策
宏观经济周期
利率、汇率、通胀等因素
国际贸易政策
补贴、税收、环保要求

重点不是简单罗列政策，而是判断：

政策会扩大还是压缩行业的盈利空间？

5. 近期新闻与重大事件

检索近期行业新闻，优先寻找能够改变行业预期的事件：

重大政策
龙头企业重大投资/扩产
并购重组
技术突破
产品价格大幅变化
重大事故/供应中断
国际事件
需求突然变化
6. 行业趋势与未来展望

把前面的信息整合起来：

行业目前处于什么周期阶段
未来 6～12 个月主要驱动因素
哪些因素可能导致景气反转
哪些细分领域可能受益
哪些细分领域可能承压
7. 风险评估

最后重点列出：

政策风险
需求下降风险
产能过剩风险
原材料价格风险
技术替代风险
国际贸易风险
行业竞争加剧风险
保证真实性和时效性
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
    ).json()
    print(message)
    PROMPT = message_format(message["choices"][0]["message"]["content"])
    (LOCAL_PATH / "PROMPT.md").write_text(PROMPT, encoding="utf-8")
    print(PROMPT)


if __name__ == "__main__":
    main()
