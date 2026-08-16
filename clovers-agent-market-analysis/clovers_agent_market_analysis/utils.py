import akshare as ak
import pandas as pd
import asyncio
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
from clovers_agent.config import CONFIG as AGENT_CONFIG
from clovers_agent import CloversAgent, Event
from clovers_agent.api import OpenAIAPI
from clovers_agent.embedding import batch_similarity
from clovers.logger import logger

WORKSPACE = Path(AGENT_CONFIG.path) / "market_analysis"
SECURITY_SYMBOL_CSV = WORKSPACE / "security_symbol.csv"


class CacheDict[K, V]:
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._cache = OrderedDict[K, V]()

    def __contains__(self, key: K):
        return key in self._cache

    def __getitem__(self, key: K):
        if key not in self._cache:
            raise KeyError(key)
        self._cache.move_to_end(key)
        return self._cache[key]

    def __setitem__(self, key: K, value: V):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()


def fmt_large_num(val):
    if pd.isna(val):
        return ""
    if abs(val) >= 100000:
        return f"{val:.2e}"
    return f"{val:.2f}"


def ohlc2md(ohlc: pd.DataFrame):
    for col in ["volume", "amount"]:
        if col in ohlc.columns:
            ohlc[col] = ohlc[col].map(fmt_large_num)
    # 1. 表头
    header = f"{ohlc.index.name or 'time'}|" + "|".join(map(str, ohlc.columns))
    # 2. 最简分界线
    divider = "|".join("-" * (len(ohlc.columns) + 1))
    # 3. 数据行
    rows = ["|".join(map(str, row)) for row in ohlc.itertuples(index=True)]
    return "\n".join((header, divider, *rows))


def get_market_microscopic_quotes(symbol: str):
    minute_k = ak.stock_zh_a_minute(symbol=symbol, period="1", adjust="qfq")
    minute_k = minute_k.rename(columns={"day": "datetime"})
    minute_k["datetime"] = pd.to_datetime(minute_k["datetime"])
    minute_k["volume"] = pd.to_numeric(minute_k["volume"], errors="coerce")
    minute_k["amount"] = pd.to_numeric(minute_k["amount"], errors="coerce")
    minute_k = minute_k.sort_values("datetime")
    start_3d = pd.Timestamp(minute_k["datetime"].dt.date.unique()[-3])
    minute_k = minute_k.set_index("datetime")
    report = ["### 15分钟实时"]
    report.append(ohlc2md(minute_k.tail(15).rename(index=lambda x: x.strftime("%Y-%m-%d %H:%M"))))
    report.append("### 3日")
    report.append(
        ohlc2md(
            minute_k.loc[start_3d:]
            .resample("1h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"})
            .dropna(subset=["open"])
            .rename(index=lambda x: x.strftime("%Y-%m-%d %H:00"))
        )
    )
    return "\n\n".join(report)


def get_market_macroscopic_quotes(symbol: str):
    hourly_k = ak.stock_zh_a_minute(symbol=symbol, period="60", adjust="qfq")
    hourly_k = hourly_k.rename(columns={"day": "date"})
    hourly_k["date"] = pd.to_datetime(hourly_k["date"])
    hourly_k["volume"] = pd.to_numeric(hourly_k["volume"], errors="coerce")
    hourly_k["amount"] = pd.to_numeric(hourly_k["amount"], errors="coerce")
    hourly_k = hourly_k.sort_values("date")
    dates = hourly_k["date"].dt.date.unique()
    start_15d = pd.Timestamp(dates[-15])
    now = datetime.now()
    start_1y = pd.Timestamp(now.replace(year=now.year - 1))
    hourly_k = hourly_k.set_index("date")
    report = ["### 15日"]
    report.append(
        ohlc2md(
            hourly_k.loc[start_15d:]
            .resample("D")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"})
            .dropna(subset=["open"])
            .rename(index=lambda x: x.strftime("%Y-%m-%d"))
        )
    )
    report.append("### 1年")
    report.append(
        ohlc2md(
            hourly_k.loc[start_1y:]
            .resample("ME")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"})
            .dropna(subset=["open"])
            .rename(index=lambda x: x.strftime("%Y-%m"))
        )
    )
    return "\n\n".join(report)


async def get_market_quotes(symbol: str):
    """获取股票行情
    Args:
        symbol (str): 股票代码，需要带市场前缀，如 sz302132
    Returns:
        str: 包含15分钟实时，3日小时K，15日日K,1年月K的行情报告
    """
    symbol = symbol.lower()
    # ak 是同步的，这里转成异步避免阻塞主循环，不并行执行防止 429
    micro = await asyncio.to_thread(get_market_microscopic_quotes, symbol)
    macro = await asyncio.to_thread(get_market_macroscopic_quotes, symbol)
    return f"{micro}\n\n{macro}"


def fmt_stock_code(symbol: str) -> str:
    """
    将股票代码补齐成6位
    """
    symbol = "".join(x for x in symbol if x in "0123456789")
    symbol_l = len(symbol)
    if symbol_l < 6:
        return symbol.zfill(6)
    elif symbol_l > 6:
        raise ValueError(f"股票代码长度不能超过6位: {symbol}")
    else:
        return symbol


def fmt_stock_prefix(symbol: str) -> str:
    """
    将股票代码格式化为带交易所前缀的代码。

    Args:
        symbol (str): 纯数字股票代码，如 "600000"

    Returns:
        str: 带前缀的代码，如 "sh600000"

    Examples:
        >>> format_stock_code("600000")
        'sh600000'
        >>> format_stock_code("000001")
        'sz000001'
        >>> format_stock_code("920000")
        'bj920000'
    """
    if symbol.startswith(("000", "001", "002", "003", "300", "301", "302")):
        return "sz"
    elif symbol.startswith(("600", "601", "603", "605", "688", "689")):
        return "sh"
    elif symbol.startswith("920"):
        return "bj"
    elif symbol.startswith(("0", "3")):
        return "sz"
    elif symbol.startswith("6"):
        return "sh"
    elif symbol.startswith(("9")):
        return "bj"
    else:
        raise ValueError(f"无法识别的股票代码: {symbol}")


def fmt_stock_symbol(symbol: str) -> str:
    symbol = fmt_stock_code(symbol)
    return f"{fmt_stock_prefix(symbol)}{symbol}"


async def update_security_symbol_data():
    """
    更新股票代码数据
    """
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    stock_codes = await asyncio.to_thread(ak.stock_info_a_code_name)
    stock_codes = stock_codes.rename(columns={"code": "symbol"})
    stock_codes["name"] = stock_codes["name"].astype(str).str.replace(" ", "")
    stock_codes["symbol"] = stock_codes["symbol"].map(fmt_stock_symbol)
    stock_codes = stock_codes.dropna(subset=["symbol"])
    index_codes = await asyncio.to_thread(ak.stock_zh_index_spot_sina)
    index_codes = index_codes.rename(columns={"代码": "symbol", "名称": "name"})[["symbol", "name"]]
    index_codes["name"] = index_codes["name"].astype(str).str.replace(" ", "")
    symbol_data = pd.concat([stock_codes, index_codes], ignore_index=True)
    symbol_data.to_csv(SECURITY_SYMBOL_CSV, index=False, encoding="utf-8")
    return symbol_data


QUERY_SYMBOL_CACHE = CacheDict[str, str](50)


async def query_security_symbol(column: str, value: str, agent: CloversAgent, limit: int = 10):
    """
    根据股票名称或代码查询股票信息
    """
    if column == "symbol":

        def query_fn_symbol(df: pd.DataFrame):
            return df[df["symbol"] == value]

        query_fn = query_fn_symbol
    elif column == "name":

        def query_fn_name(df: pd.DataFrame):
            result = df[df["name"] == value]
            if not result.empty:
                return result
            result = df[df["name"].astype(str).str.contains(value, regex=False)]
            if not result.empty:
                return result
            names = df["name"].astype(str).tolist()
            scores = batch_similarity(names, value, agent.sentence_model)
            df["similarity"] = scores
            return df.sort_values(by="similarity", ascending=False).head(limit)

        query_fn = query_fn_name

    else:
        return None
    cache_key = f"{column}:{value}"
    if cache_key in QUERY_SYMBOL_CACHE:
        return [QUERY_SYMBOL_CACHE[cache_key]]
    if not SECURITY_SYMBOL_CSV.exists():
        stock_codes = await update_security_symbol_data()
    else:
        stock_codes = pd.read_csv(SECURITY_SYMBOL_CSV, dtype=str, encoding="utf-8")
    result = query_fn(stock_codes)
    if result.empty:
        now = datetime.now()
        today_9am = now.replace(hour=9, minute=0, second=0, microsecond=0).timestamp()
        yest_15pm = today_9am - 18 * 3600
        st_mtime = SECURITY_SYMBOL_CSV.stat().st_mtime
        if st_mtime < yest_15pm or (st_mtime < today_9am and now.timestamp() > today_9am):
            stock_codes = await update_security_symbol_data()
            result = query_fn(stock_codes)
            if result.empty:
                return None
        else:
            return None
    infos = []
    for info in result.to_dict(orient="records"):
        symbol = info["symbol"]
        name = info["name"]
        item = f"{name} {symbol}"
        QUERY_SYMBOL_CACHE[f"symbol:{symbol}"] = item
        QUERY_SYMBOL_CACHE[f"name:{name}"] = item
        infos.append(item)
    return infos


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

STOCK_NEWS_RESEARCH_PROMPT = """\
你是一位资深的证券基本面分析师，请你针对用户提供的股票名称，利用搜索工具 `web_search` 和网页查看工具 `web_extractor` 获取该股票和其行业的相关新闻，并撰写一份专业基本面分析

为了保证分析的准确性与时效性，请按以下步骤地执行数据检索与分析：

1. **财务与经营状况分析**：
   - 检索最近 1-4 个季度的财报数据：营业收入及其同比增长率、净利润及其同比增长率、毛利率、净利率、ROE（净资产收益率）及经营现金流
   - 关注近期重要公告：如业绩预告、股东增减持计划、高管变动、股权激励等
2. **核心业务与商业模式拆解**：
   - 梳理主营业务构成：识别核心利润来源（盈利增长点）与拖累业绩的板块（亏损点）
   - 探究公司的竞争壁垒（如技术、渠道、成本优势）以及未来战略布局（新增产能、新市场开拓等）
3. **行业格局与新闻动态追踪**：
   - 检索该行业近期变动、上下游产业链供需变化及竞争格局
   - 收集与该公司直接相关的重大新闻、研报要点，评估相关事件对公司短期及中长期业务的影响
4. **风险因素与综合评估整合**：
   - 结合定量财务指标与定性新闻分析，识别潜在商业与财务风险
   - 总结信息，得出客观、中立的综合评价

请按照以下标准结构输出分析报告：

### 最新财报与经营状况摘要
- **业绩表现**：营业收入、净利润及其同比/环比变动
- **经营亮点**：如毛利率提升、主营业务爆发、经营性现金流改善等
- **经营隐忧**：如费用率上升、应收账款增加、存货高企或大股东减持等

### 主营业务与战略布局
- **主要盈利点**：公司靠什么赚取核心利润
- **潜在拖累项**：是否存在亏损业务或成本上升压力
- **未来动向**：公司新产品/新产能/新市场的推进情况

### 行业动态与外部影响
- 分点列举近期行业重磅新闻或政策走向
- **影响分析**：逐条简述上述行业变化对该公司业务或估值的潜在影响

### 风险评估
- **行业/政策风险**：如政策收紧、行业产能过剩
- **财务/经营风险**：如债务压力、客户集中度过高、原材料价格波动
- **市场/情绪风险**：大股东减持、解禁压力等

### 基本面综合评价
- 对公司目前的竞争地位、盈利质量与成长性给出简短明确的总结（不作直接买卖推荐，保持客观中立）

---

## 输出规范与原则
1. 时效性：使用最新的财务数据。
2. 准确性：如遇到矛盾信息请谨慎处理。
"""

INDEX_NEWS_RESEARCH_PROMPT = """\
你是一位资深的金融市场研究员。请你针对用户提供的指数名称，利用搜索工具 `web_search` 和网页查看工具 `web_extractor` 进行新闻收集并撰写一份专业的新闻分析

为了保证信息的真实性和深度，不要仅围绕指数名称本身搜索新闻。请按以下步骤地执行数据检索与分析:

1. 搜索该指数成分股中占比最大的 3-5 个行业的最新行业新闻
2. 对宏观经济环境进行分析，如相关市场的货币政策、流动性情况及关键经济指标
3. 查找是否有针对相关行业或整体市场的最新法律法规、扶持政策或限制措施
4. 识别市场近期情绪主要驱动因素

请按照以下标准结构输出分析报告：

### 综合判断
简要概述指数当前的整体态势

### 主要利好因素
- 详细描述及其对指数的支撑逻辑
- ...

### 主要利空因素
- 详细描述及其潜在的压制作用
- ...

### 影响与评估
列出需要重点关注的事件并简要解释可能的影响
列出不确定性和风险

---

## 输出规范与原则
1. 时效性：使用最新的财务数据
2. 准确性：如遇到矛盾信息请谨慎处理
"""
FUTURES_NEWS_RESEARCH_PROMPT = f"""\
你是一位资深的金融市场研究员。请你针对用户提供的期货品种，利用搜索工具 `web_search` 和网页查看工具 `web_extractor` 进行新闻收集并撰写一份清晰的新闻分析。

请按以下维度分析：

1. **供需与库存分析**：
   - 搜集最新的产量、进出口、开工率、产能利用率等数据
   - 主要贸易中心或仓库的库存变动
   - 分析下游消费终端的需求强度及订单情况

2. **宏观、政策与产业链分析**：
   - 关注国内外宏观经济数据
   - 梳理最新的行业政策、进出口关税、环保限产等政策导向
   - 分析产业链上下游的价格传导机制及利润分配现状

3. **价格驱动因素与风险综合评估**：
   - 总结当前支撑价格或压制价格的核心驱动逻辑
   - 识别潜在的风险点
   - 对短期内的市场趋势给出客观的研判

---

## 输出规范与原则
1. 时效性：使用最新的财务数据
2. 准确性：如遇到矛盾信息请谨慎处理
"""

INDUSTRY_NEWS_RESEARCH_PROMPT = """
你是一位资深的金融市场研究员。请你针对用户提供的行业方向，利用搜索工具 `web_search` 和网页查看工具 `web_extractor` 进行新闻收集并撰写一份专业的行业新闻分析

请按以下维度分析：

1. 行业定义与基本面
    - 行业边界，主要产品或服务
    - 产业链，原材料与供应商、下游客户与典型应用场景
2. 供需与景气度
    - 分析产能、产量及库存状态
    - 关注高频指标：产品价格变化、工厂开工率、订单量等
    - **重点判断**：行业景气度由需求驱动还是供给收缩驱动？
3. 政策与宏观环境
    - 罗列最新的行业政策、监管措施及扶持/限制方案
    - 考虑宏观因素：利率、汇率、通胀以及国际贸易政策的影响
    - **重点判断**：这些政策因素是会扩大还是压缩行业的整体盈利空间？
4. 风险评估
    - 从多个维度进行风险解读
"""


async def get_security_news(api: OpenAIAPI, usage_counter: dict, agent: CloversAgent, event: Event, name: str, asset_type: str):
    match asset_type:
        case "stock":
            system_prompt = STOCK_NEWS_RESEARCH_PROMPT
        case "index":
            system_prompt = INDEX_NEWS_RESEARCH_PROMPT
        case "futures":
            system_prompt = FUTURES_NEWS_RESEARCH_PROMPT
        case "industry":
            system_prompt = INDUSTRY_NEWS_RESEARCH_PROMPT
        case _:
            raise ValueError("Invalid asset type")
    today = datetime.now().strftime("%Y年%m月%d日")
    payload = api.build_payload(({"role": "user", "content": f"请根据{today}最新信息，为 {name} 撰写一份详细新闻报告"},), system_prompt)
    payload["tools"] = [agent.manifest["web_search"], agent.manifest["web_extractor"]]
    try:
        return await agent.call_turn(api, payload, usage_counter, event)
    except Exception as e:
        logger.error(f"分析 {name} 时发生错误: {e}")
        return ""


STOCK_ANALYSIS_PROMPT = """\
你是一位资深的金融分析师和投资策略专家，你的任务是根据用户提供的行情数据和相关咨询信息，为用户撰写一份专业、严谨且具有操作参考价值的股票投资报告书。

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


async def analyze_stock(api: OpenAIAPI, usage_counter: dict, agent: CloversAgent, event: Event, report: str):
    payload = api.build_payload(({"role": "user", "content": report},), STOCK_ANALYSIS_PROMPT)
    try:
        return await agent.call_turn(api, payload, usage_counter, event)
    except Exception as e:
        logger.error(f"分析股票时发生错误: {e}")
        return report
