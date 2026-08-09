import json
import asyncio
from pathlib import Path
from datetime import datetime
from clovers_agent import CloversAgent, Event, SkillCore
from clovers_agent.api import OpenAIAPI
from clovers_agent.config import CONFIG as AGENT_CONFIG
from clovers.logger import logger
from .utils import get_quotes_md

WORKSPACE = Path(AGENT_CONFIG.path) / "投资建议"

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
你是一位资深的金融分析师和投资策略专家，擅长结合基本面分析、技术指标和市场情绪来制定精确的投资计划。

你的任务是对根据用户提供的股票报告

### 任务步骤：

**1. 实时数据收集**
请使用搜索工具 `web_search` 与网页查看工具 `web_extractor` 获取该股票最新的以下信息：
- **市场概况**：当前股价、日内涨跌幅、成交量、总市值。
- **财务基本面**：市盈率 (P/E)、市净率 (P/B)、EPS（每股收益）、最新的季度收益报告摘要。
- **近期新闻**：过去 7-14 天内影响股价的重大新闻、公告或行业动态。
- **技术指标**：支撑位、阻力位、RSI、移动平均线状态。

**2. 深度分析**
在提供报告之前，请先进行如下判断：
- 分析收集到的数据是否相互矛盾（例如：利好新闻但股价下跌）。
- 评估该股票当前的估值水平（高估、合理还是低估）。
- 识别潜在的风险点（政策、财报暴雷、大盘系统性风险等）。
- 计算合理的入场区间、预期回报目标以及必须严格执行的止损位。

**3. 生成正式报告**
报告必须包含以下板块：

#### 第一部分：基本面评估
- 简述公司当前的市场地位、主营业务及行业竞争力。
- 总结财务状况的优劣。
- 分析近期新闻对股价的中长期影响。

#### 第二部分：技术面评估
- 描述当前趋势（上涨趋势、横盘整理或下跌趋势）。
- 指出关键的支撑和阻力水平。

#### 第三部分：操作方案（核心部分）
- **入场建议**：建议的具体买入价位或区间，并说明理由。
- **上行目标**：短期及中期的获利目标位。
- **止损位**：基于技术分析或风险控制的卖出点位。

#### 第四部分：风险提示
- 列出投资者需要警惕的具体风险因素。
"""


async def get_news_md(api: OpenAIAPI, usage_counter: dict, agent: CloversAgent, event: Event, symbol: str):
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
    payload = api.build_payload(
        ({"role": "user", "content": report},),
        STOCK_ANALYSIS_PROMPT,
    )
    try:
        return await agent.call_turn(api, payload, usage_counter, event)
    except Exception as e:
        return ""


TOOLS = SkillCore()


@TOOLS.create_category("investment_advice", "当用户询问投资建议时调用此工具。")
async def _(agent: CloversAgent, event: Event):
    return "若用户希望分析特定股票，在执行任务之前请先和用户确认正式名称和股票代码。"


@TOOLS.register(
    "screen_stocks_entry",
    "扫描整个股票市场，根据技术面或基本面策略筛选出当前具备入场（买入/建仓）信号的股票列表。当用户要求推荐适合入场的股票时调用此工具。",
    category="investment_advice",
)
async def _(agent: CloversAgent, event: Event):
    return "此功能暂不开放。"
    session = agent.current_session(event)
    api = session.api
    counter = session.usage_counter
    today, now = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S").split(" ")
    payload = api.build_payload(
        ({"role": "user", "content": f"请检索并整理{today}当下最值得关注的 20 支股票。"},),
        STOCK_SCREENING_PROMPT,
    )
    payload["tools"] = [agent.manifest["web_search"], agent.manifest["web_extractor"]]
    payload["response_format"] = {"type": "json_object"}
    stocks = json.loads(await agent.call_turn(api, payload, counter, event))
    logger.info(f"筛选出的股票列表: {stocks}")
    tasks = [asyncio.create_task(analyze_stock(api, counter, agent, event, stock)) for stock in stocks]
    reports = await asyncio.gather(*tasks)
    report = "\n---\n".join(reports)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    report_file = WORKSPACE / "股票入场报告.md"
    report_file.write_text(f"# 股市入场报告\n\n报告生成时间: {today} {now}\n---\n{report}", encoding="utf-8")
    await event.send("file", report_file)
    return report


@TOOLS.register(
    "analyze_stock",
    "分析指定股票，形成一个详细的报告和投资建议。当用户要求分析某个股票时调用此工具。",
    {"symbol": {"type": "string", "description": "带交易所前缀的股票代码，如：sh600000"}},
    category="investment_advice",
)
async def _(agent: CloversAgent, event: Event, symbol: str):
    session = agent.current_session(event)
    quotes = await get_quotes_md(symbol)
    news = await get_news_md(session.api, session.usage_counter, agent, event, symbol)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    report_file = WORKSPACE / f"{symbol}.md"
    report = f"{quotes}\n\n{news}"
    report_file.write_text(report, encoding="utf-8")
    await event.send("text", report)
    return "该工具暂未实现。"
