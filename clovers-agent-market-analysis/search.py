import asyncio
import akshare as ak
import pandas as pd
import numpy as np
import time


def relative_features_md(
    stock_daily_k: pd.DataFrame,
    index_daily_k: pd.DataFrame,
    excess_windows=(5, 20, 60),  # 超额收益周期
    rs_window=10,  # RS 窗口
    corr_window=60,  # 滚动相关性窗口
    capture_window=60,  # 上涨/下跌捕获率滚动窗口
):

    df = pd.concat(
        [stock_daily_k["close"].rename("stock_close"), index_daily_k["close"].rename("index_close")],
        axis=1,
        join="inner",
    ).dropna()
    stock_ret = df["stock_ret"] = df["stock_close"].pct_change()
    index_ret = df["index_ret"] = df["index_close"].pct_change()
    df["excess_ret"] = stock_ret - index_ret
    excess_ret = df[["excess_ret"]]  # excess_ret 算 Sharpe 需要保留更多数据
    length = min(len(df), 2 * max(excess_windows[-1], rs_window, corr_window, capture_window))
    df = df.tail(length)
    for x in excess_windows:
        stock_ret = df["stock_close"].pct_change(x)
        index_ret = df["index_close"].pct_change(x)
        df[f"relative_return_{x}d"] = (1 + stock_ret) / (1 + index_ret) - 1
    # RS 移动平均值、RS 变化、RS 均线斜率
    rs = df["stock_close"] / df["index_close"]

    def calc_slope(x):
        y = x.to_numpy()
        t = np.arange(len(y))
        return np.polyfit(t, y, 1)[0]

    df["rs_slope"] = pd.Series(np.log(rs)).rolling(rs_window).apply(calc_slope, raw=False)
    stock_ret = df["stock_ret"]
    index_ret = df["index_ret"]
    df["rolling_corr"] = stock_ret.rolling(corr_window).corr(index_ret)
    up_stock = (1 + stock_ret.where(index_ret > 0, 0)).rolling(capture_window).apply(np.prod, raw=True)
    up_index = (1 + index_ret.where(index_ret > 0, 0)).rolling(capture_window).apply(np.prod, raw=True)
    down_stock = (1 + stock_ret.where(index_ret < 0, 0)).rolling(capture_window).apply(np.prod, raw=True)
    down_index = (1 + index_ret.where(index_ret < 0, 0)).rolling(capture_window).apply(np.prod, raw=True)
    df["upside_capture"] = (up_stock - 1).div((up_index - 1).replace(0, np.nan))
    df["downside_capture"] = (down_stock - 1).div((down_index - 1).replace(0, np.nan))
    # 10. 相对收益 Sharpe：全期超额收益年化夏普
    now = df.index[-1]
    last_y = now - pd.DateOffset(years=1)
    excess_ret = excess_ret[last_y:]["excess_ret"]
    excess_std = excess_ret.std()
    if excess_std == 0 or np.isnan(excess_std):
        relative_sharpe = np.nan
    else:
        relative_sharpe = excess_ret.mean() / excess_std * np.sqrt(252)
    last = df.iloc[-1]
    # 12. 相对收益胜率
    report = [f"{x}D 相对收益率:{last[f"relative_return_{x}d"]:.2%}" for x in excess_windows]
    report.append(f"{rs_window}D RS对数斜率: {last["rs_slope"]:.3g}")
    report.append(f"{corr_window}D 滚动相关性: {last['rolling_corr']:.2%}")
    report.append(f"{capture_window}D 上涨捕获率: {last['upside_capture']:.2%}")
    report.append(f"{capture_window}D 下跌捕获率: {last['downside_capture']:.2%}")
    report.append(f"近{length}日跑赢指数交易日占比: {(df["excess_ret"] > 0).mean():.2%}")
    report.append(f"相对信息比率: {relative_sharpe:.2f}")
    return "\n".join(report)


async def fetch_quotes_ohlc(symbol: str, period: str, adjust: str = ""):
    ohlc = await asyncio.to_thread(ak.stock_zh_a_minute, symbol=symbol, period=period, adjust=adjust)
    ohlc = ohlc.rename(columns={"day": "datetime"})
    ohlc["datetime"] = pd.to_datetime(ohlc["datetime"])
    ohlc["open"] = pd.to_numeric(ohlc["open"], errors="coerce")
    ohlc["close"] = pd.to_numeric(ohlc["close"], errors="coerce")
    ohlc["high"] = pd.to_numeric(ohlc["high"], errors="coerce")
    ohlc["low"] = pd.to_numeric(ohlc["low"], errors="coerce")
    ohlc["volume"] = pd.to_numeric(ohlc["volume"], errors="coerce")
    ohlc["amount"] = pd.to_numeric(ohlc["amount"], errors="coerce")
    return ohlc.dropna(subset=["open", "high", "low", "close"]).sort_values("datetime").set_index("datetime")


def resample_ohlc(df: pd.DataFrame, period: str):
    return (
        df.resample(period)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum"})
        .dropna(subset=["open"])
    )


def main():
    index_hourly_k = asyncio.run(fetch_quotes_ohlc(symbol="sh000300", period="60", adjust="qfq"))
    index_daily_k = resample_ohlc(index_hourly_k, "D")
    while True:
        symbol = input("输入股票代码: ")
        stock_hourly_k = asyncio.run(fetch_quotes_ohlc(symbol=symbol, period="60", adjust="qfq"))
        stock_daily_k = resample_ohlc(stock_hourly_k, "D")
        md = relative_features_md(stock_daily_k, index_daily_k)
        print(md)


if __name__ == "__main__":
    main()
