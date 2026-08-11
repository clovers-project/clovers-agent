from clovers_agent import CloversAgent, Event, SkillCore
from .utils import WORKSPACE, query_stock_symbol, get_stock_quotes, get_stock_news, analyze_stock

TOOLS = SkillCore()


@TOOLS.create_category(
    "stock_market_analysis",
    "包含选股扫描、个股分析、实时行情，新闻分析等功能。",
)
async def _(agent: CloversAgent, event: Event):
    return """\
本工具组下所有 symbol 字段都为带交易所前缀的股票代码，如：sh600000
本工具组并发时极易造成 429 错误。在针对特定股票进行工具调用前，请使用 `query_stock_symbol` 或直接向用户确认股票正式名称和代码。
"""


@TOOLS.register(
    "query_stock_symbol",
    "根据股票名称或代码查询该股票的正式名称和代码。",
    {
        "column": {
            "type": "string",
            "description": "查询列，可选 'symbol'（按股票代码精确查询）或 'name'（按名称匹配，可能返回多个结果）。",
            "enum": ["symbol", "name"],
        },
        "value": {"type": "string", "description": "需要查询的股票名称或代码"},
    },
    category="stock_market_analysis",
)
async def _(agent: CloversAgent, event: Event, column: str, value: str):
    lines = await query_stock_symbol(column, value, agent)
    return "\n".join(lines) if lines else "未查询到结果"


# @TOOLS.register(
#     "screen_stocks_entry",
#     "扫描整个股票市场，根据技术面或基本面策略筛选出当前具备入场（买入/建仓）信号的股票列表。当用户要求推荐适合入场的股票时调用此工具。",
#     category="stock_market_analysis",
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
    {"symbol": {"type": "string", "description": "股票代码"}},
    category="stock_market_analysis",
)
async def _(agent: CloversAgent, event: Event, symbol: str):
    session = agent.current_session(event)
    quotes = await get_stock_quotes(symbol)
    news = await get_stock_news(session.api, session.usage_counter, agent, event, symbol)
    if not news:
        return "报告生成失败。"
    report = f"{quotes}\n\n{news}"
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / f"{symbol}_report.md").write_text(report, encoding="utf-8")
    advice = await analyze_stock(session.api, session.usage_counter, agent, event, report)
    if not advice:
        return "报告生成失败。"
    advice_file = WORKSPACE / f"{symbol}_advice.md"
    (advice_file).write_text(advice, encoding="utf-8")
    await event.send("file", advice_file)
    return advice


@TOOLS.register(
    "get_stock_qoutes",
    "获取指定股票实时行情",
    {"symbol": {"type": "string", "description": "股票代码"}},
    category="stock_market_analysis",
)
async def _(agent: CloversAgent, event: Event, symbol: str):
    return await get_stock_quotes(symbol) or "没有找到该股票行情"


@TOOLS.register(
    "get_stock_news",
    "获取指定股票其公司及所属行业新闻分析。",
    {"symbol": {"type": "string", "description": "股票代码"}},
    category="stock_market_analysis",
)
async def _(agent: CloversAgent, event: Event, symbol: str):
    session = agent.current_session(event)
    news = await get_stock_news(session.api, session.usage_counter, agent, event, symbol)
    if not news:
        return "无法获取股票新闻。"
    return news
