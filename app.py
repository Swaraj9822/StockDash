import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

# --------------------------
# Page config
# --------------------------
st.set_page_config(
    page_title="NIFTY + VIX Live Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------
# Sidebar controls
# --------------------------
st.sidebar.title("Controls")
refresh_seconds = st.sidebar.slider("Auto-refresh every (seconds)", 5, 120, 30, 5)
default_symbol = "^NSEI"  # NIFTY 50 on Yahoo Finance
vix_symbol = "^INDIAVIX"  # India VIX on Yahoo Finance

interval = st.sidebar.selectbox("Intraday interval", ["1m", "2m", "5m", "15m"], index=0)
period = st.sidebar.selectbox("Period", ["1d", "5d"], index=0)

show_ema = st.sidebar.checkbox("Show EMA(20, 50)", True)
show_sma = st.sidebar.checkbox("Show SMA(200)", True)
show_bbands = st.sidebar.checkbox("Show Bollinger Bands(20, 2)", True)
show_macd = st.sidebar.checkbox("Show MACD(12, 26, 9)", True)

# Auto-refresh
st_autorefresh(interval=refresh_seconds * 1000, key="autorefresh")

# --------------------------
# Data fetch helpers
# --------------------------
def safe_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Try primary intraday request first; if empty, fall back gracefully to broader period/intervals.
    """
    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        prepost=False,
        threads=True,
        progress=False,
    )
    if df is None or df.empty:
        # Fallbacks if intraday not available
        fallbacks = [
            ("5d", "5m"),
            ("1mo", "1d"),
        ]
        for p, i in fallbacks:
            dff = yf.download(
                tickers=ticker,
                period=p,
                interval=i,
                auto_adjust=False,
                prepost=False,
                threads=True,
                progress=False,
            )
            if dff is not None and not dff.empty:
                return dff
        return pd.DataFrame()
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Ensure standard OHLC names
    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]

    # RSI
    out["RSI_14"] = ta.rsi(out["Close"], length=14)

    # Moving averages
    if show_ema:
        out["EMA_20"] = ta.ema(out["Close"], length=20)
        out["EMA_50"] = ta.ema(out["Close"], length=50)
    if show_sma:
        out["SMA_200"] = ta.sma(out["Close"], length=200)

    # Bollinger Bands
    if show_bbands:
        bb = ta.bbands(out["Close"], length=20, std=2)
        if bb is not None and not bb.empty:
            out["BBL"] = bb.iloc[:, 0]
            out["BBM"] = bb.iloc[:, 1]
            out["BBU"] = bb.iloc[:, 2]

    # MACD
    if show_macd:
        macd = ta.macd(out["Close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            out["MACD"] = macd.iloc[:, 0]
            out["MACD_SIGNAL"] = macd.iloc[:, 1]
            out["MACD_HIST"] = macd.iloc[:, 2]

    return out


def latest_value(series: pd.Series):
    try:
        return float(series.dropna().iloc[-1])
    except Exception:
        return np.nan


# --------------------------
# Fetch data
# --------------------------
nifty_df_raw = safe_download(default_symbol, period=period, interval=interval)
vix_df_raw = safe_download(vix_symbol, period=period, interval=interval)

nifty_df = compute_indicators(nifty_df_raw) if not nifty_df_raw.empty else pd.DataFrame()
vix_df = vix_df_raw.copy()

# --------------------------
# Header and metrics
# --------------------------
st.title("NIFTY 50 + India VIX Live Dashboard")

colA, colB, colC, colD = st.columns(4)

# NIFTY metrics
if not nifty_df.empty:
    last_close = latest_value(nifty_df["Close"])
    prev_close = latest_value(nifty_df["Close"].shift(1))
    pct = ((last_close - prev_close) / prev_close * 100.0) if prev_close and prev_close > 0 else 0.0
    rsi = latest_value(nifty_df["RSI_14"])
else:
    last_close, pct, rsi = np.nan, np.nan, np.nan

with colA:
    st.metric(label="NIFTY 50 (Last)", value=f"{last_close:,.2f}" if np.isfinite(last_close) else "—",
              delta=f"{pct:+.2f}%" if np.isfinite(pct) else "—")

# India VIX metrics
if not vix_df.empty:
    vix_last = latest_value(vix_df["Close"] if "Close" in vix_df.columns else vix_df["Adj Close"])
    vix_prev = latest_value((vix_df["Close"] if "Close" in vix_df.columns else vix_df["Adj Close"]).shift(1))
    vix_pct = ((vix_last - vix_prev) / vix_prev * 100.0) if vix_prev and vix_prev > 0 else 0.0
else:
    vix_last, vix_pct = np.nan, np.nan

with colB:
    st.metric(label="India VIX (Last)", value=f"{vix_last:,.2f}" if np.isfinite(vix_last) else "—",
              delta=f"{vix_pct:+.2f}%" if np.isfinite(vix_pct) else "—")

with colC:
    st.metric(label="RSI(14)", value=f"{rsi:.1f}" if np.isfinite(rsi) else "—")

with colD:
    st.write(f"Refreshed at {datetime.now().strftime('%H:%M:%S')}")

st.caption("Data from Yahoo Finance; intraday availability and timing may vary by exchange.")

# --------------------------
# Price chart with indicators
# --------------------------
if not nifty_df.empty:
    fig = go.Figure()
    # Candles (if available intraday)
    if {"Open", "High", "Low", "Close"}.issubset(nifty_df.columns):
        fig.add_trace(go.Candlestick(
            x=nifty_df.index,
            open=nifty_df["Open"],
            high=nifty_df["High"],
            low=nifty_df["Low"],
            close=nifty_df["Close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=nifty_df.index,
            y=nifty_df["Close"],
            name="Close",
            mode="lines"
        ))

    # EMA/SMA
    if show_ema and "EMA_20" in nifty_df.columns:
        fig.add_trace(go.Scatter(x=nifty_df.index, y=nifty_df["EMA_20"], name="EMA 20", line=dict(color="#008FFB")))
    if show_ema and "EMA_50" in nifty_df.columns:
        fig.add_trace(go.Scatter(x=nifty_df.index, y=nifty_df["EMA_50"], name="EMA 50", line=dict(color="#FEB019")))
    if show_sma and "SMA_200" in nifty_df.columns:
        fig.add_trace(go.Scatter(x=nifty_df.index, y=nifty_df["SMA_200"], name="SMA 200", line=dict(color="#00E396")))

    # Bollinger Bands
    if show_bbands and {"BBL", "BBM", "BBU"}.issubset(nifty_df.columns):
        fig.add_trace(go.Scatter(x=nifty_df.index, y=nifty_df["BBU"], name="BB Upper", line=dict(color="gray", width=1)))
        fig.add_trace(go.Scatter(x=nifty_df.index, y=nifty_df["BBM"], name="BB Mid", line=dict(color="lightgray", width=1)))
        fig.add_trace(go.Scatter(x=nifty_df.index, y=nifty_df["BBL"], name="BB Lower", line=dict(color="gray", width=1), fill='tonexty', fillcolor="rgba(200,200,200,0.2)"))

    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Time",
        yaxis_title="Price (INR)",
    )
    st.subheader("NIFTY 50 Price + Indicators")
    st.plotly_chart(fig, use_container_width=True)

    # RSI
    if "RSI_14" in nifty_df.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=nifty_df.index, y=nifty_df["RSI_14"], name="RSI(14)", line=dict(color="#546E7A")))
        fig_rsi.add_hline(y=70, line_color="#ef5350", opacity=0.6)
        fig_rsi.add_hline(y=30, line_color="#26a69a", opacity=0.6)
        fig_rsi.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="RSI",
        )
        st.subheader("RSI(14)")
        st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD
    if show_macd and {"MACD", "MACD_SIGNAL", "MACD_HIST"}.issubset(nifty_df.columns):
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=nifty_df.index, y=nifty_df["MACD"], name="MACD", line=dict(color="#546E7A")))
        fig_macd.add_trace(go.Scatter(x=nifty_df.index, y=nifty_df["MACD_SIGNAL"], name="Signal", line=dict(color="#FF9800")))
        fig_macd.add_trace(go.Bar(x=nifty_df.index, y=nifty_df["MACD_HIST"], name="Hist", marker_color="#90A4AE"))
        fig_macd.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_title="MACD",
            barmode="relative"
        )
        st.subheader("MACD(12, 26, 9)")
        st.plotly_chart(fig_macd, use_container_width=True)
else:
    st.warning("No NIFTY 50 data available for the selected period/interval. Try a different combination.")
