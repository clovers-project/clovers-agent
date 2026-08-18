import pandas as pd
from datetime import datetime
from clovers_agent import CloversAgent, Event, SkillCore
from clovers.logger import logger
from .utils import (
    WORKSPACE,
    query_security_symbol,
    fetch_quotes_ohlc,
    resample_ohlc,
    realtime_ohlc_to_md,
    historical_ohlc_to_md,
    relative_features_md,
    STOCK_NEWS_RESEARCH_PROMPT,
    INDEX_NEWS_RESEARCH_PROMPT,
    FUTURES_NEWS_RESEARCH_PROMPT,
    INDUSTRY_NEWS_RESEARCH_PROMPT,
    STOCK_ANALYSIS_PROMPT,
)

TOOLS = SkillCore()
MARKET_ANALYSIS = "market_analysis"


@TOOLS.create_category(MARKET_ANALYSIS, "个股/指数/期货等相关功能。包含参数计算、实时行情，新闻分析等功能。")
async def _(agent: CloversAgent, event: Event):
    return """\
## 介绍
本工具组下所有带 symbol 的字段都为带交易所前缀的代码，如：`sh000001`，`sz000001`，仅支持A股的股票与指数。
`ref_index_symbol` 为参考指数代码，默认为 `sh000300`。除非用户指定，否则字段为空。
若不知道股票正式名称或代码，请使用 `query_security_symbol`查询或直接向用户确认，严禁猜测或捏造。

## 下面的工具在一次任务中只能执行一次
- 分析特定金融对象时使用 `analyze_security`
- 分析多个股票时只比较相对强弱，使用 `compute_stock_relative_features` 
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
    "fetch_realtime_quotes",
    "获取指定股票或指数实时行情",
    {"symbol": {"type": "string", "description": "股票或指数代码"}},
    category=MARKET_ANALYSIS,
)
async def _(agent: CloversAgent, event: Event, symbol: str):
    return realtime_ohlc_to_md(await fetch_quotes_ohlc(symbol, "15"))


@TOOLS.register(
    "analyze_security",
    "获取指定金融对象的分析报告",
    {
        "name": {"type": "string", "description": "股票/指数/期货/行业名称"},
        "asset_type": {
            "type": "string",
            "description": "金融对象类型，用于选择相应的新闻分析策略。",
            "enum": ["stock", "index", "futures", "industry"],
        },
        "ref_index_symbol": {"type": "string", "description": "参考指数代码，仅当 `asset_type` 为 `stock` 时有效。"},
    },
    category=MARKET_ANALYSIS,
    required=["name", "asset_type"],
)
async def _(agent: CloversAgent, event: Event, name: str, asset_type: str, ref_index_symbol: str = "sh000300"):
    match asset_type:
        case "stock":
            stocks_info = await query_security_symbol("name", name, agent)
            if not stocks_info:
                return f"未找到{name}"
            if len(stocks_info) > 1:
                return f"{name} 匹配到多个结果:\n{'\n'.join(stocks_info)}"
            name = stocks_info[0]
            system_prompt = STOCK_NEWS_RESEARCH_PROMPT
        case "index":
            system_prompt = INDEX_NEWS_RESEARCH_PROMPT
        case "futures":
            system_prompt = FUTURES_NEWS_RESEARCH_PROMPT
        case "industry":
            system_prompt = INDUSTRY_NEWS_RESEARCH_PROMPT
        case _:
            return "无效的资产类型"
    session = agent.current_session(event)
    user_prompt = f"请根据{datetime.now().strftime("%Y年%m月%d日")}最新信息，为 {name} 撰写一份详细新闻报告"
    payload = session.api.build_payload(({"role": "user", "content": user_prompt},), system_prompt)
    payload["tools"] = [agent.manifest["web_search"], agent.manifest["web_extractor"]]
    news = await agent.call_turn(session.api, payload, session.usage_counter, event)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    news_file = WORKSPACE / f"{name}_{asset_type}.md"
    news_file.write_text(news, encoding="utf-8")
    if asset_type != "stock":
        if coro := event.send("file", news_file):
            await coro
        return news
    stock_info = name
    name, stock_symbol = stock_info.split(" ")
    index_info = await query_security_symbol("symbol", ref_index_symbol, agent)
    if not index_info or len(index_info) > 1:
        index_info = "沪深300 sh000300"
        ref_index_symbol = "sh000300"
    else:
        index_info = index_info[0]
    ref_index_symbol = ref_index_symbol.lower()
    # ak 是同步的，这里转成异步避免阻塞主循环，不并行执行防止 429
    stock_hourly_k = await fetch_quotes_ohlc(symbol=stock_symbol, period="60", adjust="qfq")
    index_hourly_k = await fetch_quotes_ohlc(symbol=ref_index_symbol, period="60", adjust="qfq")
    stock_daily_k = resample_ohlc(stock_hourly_k, "D")
    index_daily_k = resample_ohlc(index_hourly_k, "D")
    if stock_hourly_k.index[-1] > pd.Timestamp.now().normalize():
        current = ""
    else:
        current = realtime_ohlc_to_md(await fetch_quotes_ohlc(stock_symbol, "15")) + "\n\n"
    report = f"""
## {stock_info}

{current}{historical_ohlc_to_md(stock_hourly_k)}

## 股票相对参考指数（{index_info}）表现

{relative_features_md(stock_daily_k, index_daily_k)}

{news}
"""
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / f"{stock_symbol}_report.md").write_text(report, encoding="utf-8")
    payload = session.api.build_payload(({"role": "user", "content": report},), STOCK_ANALYSIS_PROMPT)
    try:
        advice = await agent.call_turn(session.api, payload, session.usage_counter, event)
    except Exception as e:
        logger.error(f"分析股票时发生错误: {e}")
        return report
    advice_file = WORKSPACE / f"{stock_symbol}_advice.md"
    advice_file.write_text(advice, encoding="utf-8")
    if coro := event.send("file", advice_file):
        await coro
    return advice


@TOOLS.register(
    "compute_stock_relative_features",
    "计算指定股票与参考指数的相对特征。",
    {
        "stocks": {"type": "array", "description": "待计算的股票名称或代码列表"},
        "ref_index_symbol": {"type": "string", "description": "参考指数代码"},
    },
    category=MARKET_ANALYSIS,
    required=["stocks"],
)
async def _(agent: CloversAgent, event: Event, stocks: list[str], ref_index_symbol: str = "sh000300"):
    index_info = await query_security_symbol("symbol", ref_index_symbol, agent)
    if not index_info or len(index_info) > 1:
        index_info = "沪深300 sh000300"
        ref_index_symbol = "sh000300"
    else:
        index_info = index_info[0]
    report = [f"## 参考指数：{index_info}"]
    index_hourly_k = await fetch_quotes_ohlc(symbol=ref_index_symbol, period="60", adjust="qfq")
    index_daily_k = resample_ohlc(index_hourly_k, "D")
    for stock in stocks:
        symbol = await query_security_symbol("symbol", stock, agent)
        if not symbol:
            continue
        name, symbol = symbol[0].split(" ")
        stock_hourly_k = await fetch_quotes_ohlc(symbol=symbol, period="60", adjust="qfq")
        stock_daily_k = resample_ohlc(stock_hourly_k, "D")
        report.append(f"## {name}")
        report.append(relative_features_md(stock_daily_k, index_daily_k))
    report = "\n\n".join(report)
    return report
