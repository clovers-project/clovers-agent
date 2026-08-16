import asyncio
from clovers_agent import CloversAgent, Event, SkillCore
from .utils import (
    WORKSPACE,
    query_security_symbol,
    get_market_macroscopic_quotes,
    get_market_quotes,
    get_security_news,
    analyze_stock,
)

TOOLS = SkillCore()
MARKET_ANALYSIS = "market_analysis"


@TOOLS.create_category(MARKET_ANALYSIS, "个股/指数/期货等相关功能。包含选股扫描、实时行情，新闻分析等功能。")
async def _(agent: CloversAgent, event: Event):
    return """\
本工具组下所有 symbol 字段都为带交易所前缀的代码，如：`sh000001`，`sz000001`。
本工具组并发时极易造成 429 错误。在针对特定股票进行工具调用前，请使用 `query_stock_symbol` 或直接向用户确认股票正式名称和代码。
`analyze_stock` 仅支持A股股票，`get_market_quotes` 与 `query_security_symbol` 仅支持A股的股票与指数。
对其他的分析请使用 `get_security_news`
"""


@TOOLS.register(
    "query_security_symbol",
    "根据股票或指数的名称或代码查询其正式名称和代码",
    {
        "column": {
            "type": "string",
            "description": "查询列，可选 'symbol'（按代码精确查询）或 'name'（按名称匹配，可能返回多个结果）。",
            "enum": ["symbol", "name"],
        },
        "value": {"type": "string", "description": "需要查询的股票或指数的名称或代码"},
    },
    category=MARKET_ANALYSIS,
)
async def _(agent: CloversAgent, event: Event, column: str, value: str):
    lines = await query_security_symbol(column, value, agent)
    return "\n".join(lines) if lines else "未查询到结果"


@TOOLS.register(
    "get_market_quotes",
    "获取指定股票或指数实时行情",
    {"symbol": {"type": "string", "description": "股票或指数代码"}},
    category=MARKET_ANALYSIS,
)
async def _(agent: CloversAgent, event: Event, symbol: str):
    return (await get_market_quotes(symbol)) or f"没有找到 {symbol} 的行情"


@TOOLS.register(
    "get_security_news",
    "获取指定金融对象的新闻分析。",
    {
        "name": {"type": "string", "description": "股票/指数/期货/行业名称"},
        "asset_type": {
            "type": "string",
            "description": "金融对象类型，用于选择相应的新闻分析策略。",
            "enum": ["stock", "index", "futures", "industry"],
        },
    },
    category=MARKET_ANALYSIS,
)
async def _(agent: CloversAgent, event: Event, name: str, asset_type: str):

    session = agent.current_session(event)
    news = await get_security_news(session.api, session.usage_counter, agent, event, name, asset_type)
    if not news:
        return "无法获取新闻。"
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    news_file = WORKSPACE / f"{name}_{asset_type}_news.md"
    news_file.write_text(news, encoding="utf-8")
    if coro := event.send("file", news_file):
        await coro
    return news


# @TOOLS.register(
#     "screen_stocks_entry",
#     "扫描整个股票市场，根据技术面或基本面策略筛选出当前具备入场（买入/建仓）信号的股票列表。当用户要求推荐适合入场的股票时调用此工具。",
#     category=MARKET_ANALYSIS,
# )
# async def _(agent: CloversAgent, event: Event):
#     session = agent.current_session(event)
#     api = session.api
#     counter = session.usage_counter
#     today, now = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S").split(" ")
#     payload = api.build_payload(
#         ({"role": "user", "content": f"请检索并整理{today}当下最值得关注的 20 支股票。"},),
#         STOCK_SCREENING_PROMPT,
#     )
#     payload["tools"] = [agent.manifest["web_search"], agent.manifest["web_extractor"]]
#     payload["response_format"] = {"type": "json_object"}
#     stocks = json.loads(await agent.call_turn(api, payload, counter, event))
#     logger.info(f"筛选出的股票列表: {stocks}")
#     tasks = [asyncio.create_task(analyze_stock(api, counter, agent, event, stock)) for stock in stocks]
#     reports = await asyncio.gather(*tasks)
#     report = "\n---\n".join(reports)
#     WORKSPACE.mkdir(parents=True, exist_ok=True)
#     report_file = WORKSPACE / "股票入场报告.md"
#     report_file.write_text(f"# 股市入场报告\n\n报告生成时间: {today} {now}\n---\n{report}", encoding="utf-8")
#     await event.send("file", report_file)
#     return report


@TOOLS.register(
    "analyze_stock",
    "用于分析指定股票，形成一个包含基本面，技术面，风险评估的详细报告。此工具会进行完全分析，若使用此工具则禁止调用其他工具。",
    {
        "stock_symbol": {"type": "string", "description": "股票代码"},
        "index_symbol": {"type": "string", "description": "指数代码，该指数用来参考股价行情。除非用户指定参考指数，否则此字段为空。"},
    },
    category=MARKET_ANALYSIS,
    required=["stock_symbol"],
)
async def _(agent: CloversAgent, event: Event, stock_symbol: str, index_symbol: str = "sh000300"):
    stock_info = await query_security_symbol("symbol", stock_symbol, agent)
    if not stock_info or len(stock_info) > 1:
        return "报告生成失败。"
    stock_info = stock_info[0]
    index_info = await query_security_symbol("symbol", index_symbol, agent)
    if not index_info or len(index_info) > 1:
        index_info = "沪深300 sh000300"
        index_symbol = "sh000300"
    else:
        index_info = index_info[0]
    session = agent.current_session(event)
    news = await get_security_news(session.api, session.usage_counter, agent, event, stock_info, "stock")
    if not news:
        return "报告生成失败。"
    index_quotes = await asyncio.to_thread(get_market_macroscopic_quotes, index_symbol)
    stock_quotes = await get_market_quotes(stock_symbol)
    report = f"""\
## {stock_info}

{stock_quotes}

## 参考指数：{index_info}

{index_quotes}

{news}"""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / f"{stock_symbol}_report.md").write_text(report, encoding="utf-8")
    advice = await analyze_stock(session.api, session.usage_counter, agent, event, report)
    if not advice:
        return "报告生成失败。"
    advice_file = WORKSPACE / f"{stock_symbol}_advice.md"
    advice_file.write_text(advice, encoding="utf-8")
    if coro := event.send("file", advice_file):
        await coro
    return advice
