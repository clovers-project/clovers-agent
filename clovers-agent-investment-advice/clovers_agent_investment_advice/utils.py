import akshare as ak
import pandas as pd
import asyncio
from datetime import datetime, timedelta


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


async def get_quotes_md(symbol: str):
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
