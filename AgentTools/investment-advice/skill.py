import json
from datetime import datetime
from clovers_agent import CloversAgent, Event
from clovers.logger import logger

STOCK_SCREENING_PROMPT = """\
你是一位专业的金融市场分析师，擅长追踪和分析中国 A 股市场的实时动态。
你的任务是利用搜索工具 `web_search` 获取今日最热门的 A 股股票资讯，并利用 `web_extractor` 查看资讯，多个来源整理出 100 个热点股票。

请按照以下步骤执行任务：

1. **信息检索**：调用搜索工具，搜索今日（A股最新交易日）的市场新闻、热门板块、龙头企业、涨幅榜及社交媒体讨论热点。
2. **筛选分析**：从搜索结果中筛选出 100 个具有重大新闻影响或热门讨论的A 股股票。
3. **格式化输出**：将筛选出的 100 个股票整理成一个 JSON 字符串数组。数组中的每个元素必须遵循特定的格式：`"[股票代码] 股票名称"`。

**输出要求：**

- 输出结果必须是一个标准的 JSON 数组，不包含多余的文字说明。
"""

STOCK_ANALYSIS_PROMPT = """\
你是一位专业的金融市场分析师，擅长追踪和分析中国 A 股市场的实时动态。
你的任务利用 `web_extractor` 查看资讯，对 100 个热点股票进行详细分析，并给出一个 100 个股票的详细报告。

请按照以下步骤执行任务：

1. **信息获取**：调用 `web_extractor` 查看资讯，获取 100 个热点股票的详细介绍。
"""


async def on_investment_advice(agent: CloversAgent, event: Event, content: str):
    session = agent.current_session(event)
    session.api = agent.api("investment_advice")
    session.payload = session.api.build_payload(session, f"{content}\n{agent.base_prompt}")
    session.payload["tools"] = agent.select_tools("network").copy()
    return ""


async def screen_stocks_entry(agent: CloversAgent, event: Event, content: str):
    session = agent.current_session(event)
    api = session.api
    counter = session.usage_counter
    payload = api.build_payload(
        ({"role": "user", "content": f"请立即开始执行任务，检索并整理{datetime.now().strftime('%Y-%m-%d')} 最热门的 100 个 A 股股票。"},),
        STOCK_SCREENING_PROMPT,
    )
    payload["tools"] = agent.select_tools("network").copy()
    payload["response_format"] = {"type": "json_object"}
    stocks = json.loads(await agent.call_turn(api, payload, counter, event))
    logger.info(f"筛选出的股票列表: {stocks}")
    return "接口暂时未实现"
