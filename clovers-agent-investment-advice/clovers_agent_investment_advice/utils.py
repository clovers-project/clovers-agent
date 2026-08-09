import akshare as ak
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from clovers_agent import CloversAgent, Event
from clovers_agent.api import OpenAIAPI
from clovers.logger import logger


def format_large_num(val):
    if pd.isna(val):
        return ""
    if abs(val) >= 100000:
        return f"{val:.2e}"
    return f"{val:.2f}"


def ohlc2md(ohlc: pd.DataFrame):
    for col in ["volume", "amount"]:
        if col in ohlc.columns:
            ohlc[col] = ohlc[col].map(format_large_num)
    if "turnover" in ohlc.columns:
        ohlc["turnover"] = pd.to_numeric(ohlc["turnover"], errors="coerce").map(lambda x: f"{x:.2g}")
    # 1. 表头
    header = f"{ohlc.index.name or 'time'}|" + "|".join(map(str, ohlc.columns))
    # 2. 最简分界线
    divider = "|".join("-" * (len(ohlc.columns) + 1))
    # 3. 数据行
    rows = ["|".join(map(str, row)) for row in ohlc.itertuples(index=True)]
    return "\n".join((header, divider, *rows))


async def get_stock_quotes(symbol: str):
    """获取股票行情
    Args:
        symbol (str): 股票代码，需要带市场前缀，如 sz302132
    Returns:
        str: 包含15分钟实时，3日小时K，15日日K,1年月K的行情报告
    """
    symbol = symbol.lower()
    # ak 是同步的，这里转成异步避免阻塞主循环
    minute_k = await asyncio.to_thread(ak.stock_zh_a_minute, symbol=symbol, period="1", adjust="qfq")
    minute_k = minute_k.rename(columns={"day": "datetime"})
    minute_k["datetime"] = pd.to_datetime(minute_k["datetime"])
    minute_k["volume"] = pd.to_numeric(minute_k["volume"], errors="coerce")
    minute_k["amount"] = pd.to_numeric(minute_k["amount"], errors="coerce")
    start_3d = pd.Timestamp(minute_k["datetime"].dt.date.unique()[-3])
    minute_k = minute_k.set_index("datetime")
    report = ["# 基本行情"]
    report.append("## 15分钟实时")
    report.append(ohlc2md(minute_k.tail(15).rename(index=lambda x: x.strftime("%Y-%m-%d %H:%M"))))
    report.append("## 3日")
    report.append(
        ohlc2md(
            minute_k.loc[str(start_3d) :]
            .resample("1h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"})
            .dropna(subset=["open"])
            .rename(index=lambda x: x.strftime("%Y-%m-%d %H:00"))
        )
    )
    now = datetime.now()
    daily_k = await asyncio.to_thread(
        ak.stock_zh_a_daily,
        symbol=symbol,
        start_date=(now - timedelta(days=365)).strftime("%Y%m%d"),
        end_date=now.strftime("%Y%m%d"),
        adjust="qfq",
    )
    daily_k = daily_k.drop(columns=["outstanding_share"])
    daily_k["date"] = pd.to_datetime(daily_k["date"])
    daily_k["volume"] = pd.to_numeric(daily_k["volume"], errors="coerce")
    daily_k["amount"] = pd.to_numeric(daily_k["amount"], errors="coerce")
    daily_k = daily_k.set_index("date")
    report.append("## 15日")
    report.append(ohlc2md(daily_k.tail(15).rename(index=lambda x: x.strftime("%Y-%m-%d"))))
    report.append("## 1年")
    daily_k = daily_k.drop(columns=["turnover"])
    report.append(
        ohlc2md(
            daily_k.resample("ME")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"})
            .dropna(subset=["open"])
            .rename(index=lambda x: x.strftime("%Y-%m"))
        )
    )
    return "\n\n".join(report)


STOCK_SCREENING_PROMPT = """\
你是一位专业的金融市场分析师，擅长追踪和分析中国 A 股市场的实时动态。
你的任务是利用搜索工具 `web_search` 获取今日最热门的 A 股股票资讯，并利用 `web_extractor` 查看资讯，多个来源整理出 20 个热点股票。

请按照以下步骤执行任务：

1. **信息检索**：调用搜索工具，搜索今日（A股最新交易日）的市场新闻、热门板块、龙头企业、涨幅榜及社交媒体讨论热点。
2. **筛选分析**：从搜索结果中筛选出 20 个具有重大新闻影响或热门讨论的A 股股票。
3. **格式化输出**：将筛选出的 20 个股票整理成一个 JSON 字符串数组。数组中的每个元素必须遵循特定的格式：`"[股票代码] 股票名称"`。

**输出要求：**

- 输出结果必须是一个标准的 JSON 数组，不包含多余的文字说明。
"""

STOCK_NEWS_RESERCH_PROMPT = """\
你是一位资深的证券基本面分析师，请你针对用户提供的股票代码，利用搜索工具 `web_search` 和网页查看工具 `web_extractor` 获取该股票和其行业的相关新闻，并撰写一份专业基本面分析。

为了保证分析的准确性与时效性，请按以下步骤循序渐进地执行数据检索与分析：

1. **财务与经营状况分析**：
   - 检索最近 1-4 个季度的财报数据：营业收入及其同比增长率、净利润及其同比增长率、毛利率、净利率、ROE（净资产收益率）及经营现金流。
   - 关注近期重要公告：如业绩预告、股东增减持计划、高管变动、股权激励等。
2. **核心业务与商业模式拆解**：
   - 梳理主营业务构成：识别核心利润来源（盈利增长点）与拖累业绩的板块（亏损点）。
   - 探究公司的竞争壁垒（如技术、渠道、成本优势）以及未来战略布局（新增产能、新市场开拓等）。
3. **行业格局与新闻动态追踪**：
   - 检索该行业近期变动、上下游产业链供需变化及竞争格局。
   - 收集与该公司直接相关的重大新闻、研报要点，评估相关事件对公司短期及中长期业务的影响。
4. **风险因素与综合评估整合**：
   - 结合定量财务指标与定性新闻分析，识别潜在商业与财务风险。
   - 总结信息，得出客观、中立的综合评价。

请按照以下标准结构输出分析报告：

### 最新财报与经营状况摘要
- **业绩表现**：营业收入、净利润及其同比/环比变动。
- **经营亮点**：如毛利率提升、主营业务爆发、经营性现金流改善等。
- **经营隐忧**：如费用率上升、应收账款增加、存货高企或大股东减持等。

### 主营业务与战略布局
- **主要盈利点**：公司靠什么赚取核心利润？
- **潜在拖累项**：是否存在亏损业务或成本上升压力？
- **未来动向**：公司新产品/新产能/新市场的推进情况。

### 行业动态与外部影响
- 分点列举近期行业重磅新闻或政策走向。
- **影响分析**：逐条简述上述行业变化对该公司业务或估值的潜在影响。

### 风险评估
- **行业/政策风险**：如政策收紧、行业产能过剩。
- **财务/经营风险**：如债务压力、客户集中度过高、原材料价格波动。
- **市场/情绪风险**：大股东减持、解禁压力等。

### 基本面综合评价
- 对公司目前的竞争地位、盈利质量与成长性给出简短明确的总结（不作直接买卖推荐，保持客观中立）。

---

## 输出规范与原则
1. 时效性：使用最新的财务数据。
2. 准确性：如遇到矛盾信息请谨慎处理。
"""

STOCK_ANALYSIS_PROMPT = """\
你是一位资深的金融分析师和投资策略专家，你的任务是根据用户提供的股票行情数据和相关咨询信息，为用户撰写一份专业、严谨且具有操作参考价值的股票投资建议书。

在撰写最终报告之前，请分析：
1. **基本面分析**：结合财务数据（如市盈率、毛利率等）和最新咨询（政策、财报、行业动态），评估公司的核心价值和增长潜力。
2. **技术面分析**：根据行情数据（价格走势、成交量、移动平均线等），判断当前的趋势（多头、空头或震荡）及强弱。
3. **策略制定**：基于以上分析，确定合理的入场时机、分批止盈的目标位以及防守止损位。

请按照以下结构输出你的投资建议：

### 第一部分：基本面评估
- 简述公司当前的市场地位、主营业务及行业竞争力。
- 总结财务状况的优劣。
- 分析近期新闻对股价的中长期影响。

### 第二部分：技术面评估
- 描述当前趋势（上涨趋势、横盘整理或下跌趋势）。
- 指出关键的支撑和阻力水平。

### 第三部分：操作方案
- **入场建议**：建议的具体买入价位或区间，并说明理由。
- **上行目标**：短期及中期的获利目标位。
- **止损位**：基于技术分析或风险控制的卖出点位。

### 第四部分：风险提示
- 列出投资者需要警惕的具体风险因素。
"""


async def get_stock_news(api: OpenAIAPI, usage_counter: dict, agent: CloversAgent, event: Event, symbol: str):
    today = datetime.now().strftime("%Y年%m月%d日")
    payload = api.build_payload(
        ({"role": "user", "content": f"请根据{today}最新信息，为股票代码 {symbol} 撰写一份详细的基本面报告"},),
        STOCK_NEWS_RESERCH_PROMPT,
    )
    payload["tools"] = [agent.manifest["web_search"], agent.manifest["web_extractor"]]
    try:
        return await agent.call_turn(api, payload, usage_counter, event)
    except Exception as e:
        logger.error(f"分析股票 {symbol} 时发生错误: {e}")
        return ""


async def analyze_stock(api: OpenAIAPI, usage_counter: dict, agent: CloversAgent, event: Event, report: str):
    payload = api.build_payload(({"role": "user", "content": report},), STOCK_ANALYSIS_PROMPT)
    try:
        return await agent.call_turn(api, payload, usage_counter, event)
    except Exception as e:
        logger.error(f"分析股票时发生错误: {e}")
        return report
