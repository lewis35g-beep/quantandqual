# ai_stock_forex_scanner.py

import os
import yfinance as yf
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


FOREX_PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "EURGBP", "EURAUD", "AUDJPY", "CADJPY", "CHFJPY"
]


def clean_ticker(symbol):
    symbol = symbol.upper().strip()

    if symbol in FOREX_PAIRS:
        return symbol + "=X", "forex"

    return symbol, "stock"


def calculate_rsi(data, period=14):
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(data):
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def get_market_data(symbol, asset_type):
    if asset_type == "forex":
        data = yf.download(symbol, period="6mo", interval="1h", progress=False)
    else:
        data = yf.download(symbol, period="1y", interval="1d", progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data.dropna()


def technical_analysis(data):
    data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()
    data["EMA50"] = data["Close"].ewm(span=50, adjust=False).mean()
    data["EMA200"] = data["Close"].ewm(span=200, adjust=False).mean()
    data["RSI"] = calculate_rsi(data)
    data["MACD"], data["MACD_SIGNAL"] = calculate_macd(data)

    latest = data.iloc[-1]

    price = latest["Close"]
    ema20 = latest["EMA20"]
    ema50 = latest["EMA50"]
    ema200 = latest["EMA200"]
    rsi = latest["RSI"]
    macd = latest["MACD"]
    macd_signal = latest["MACD_SIGNAL"]

    score = 0
    notes = []

    if price > ema20 > ema50:
        score += 2
        notes.append("Bullish short-term trend.")
    elif price < ema20 < ema50:
        score -= 2
        notes.append("Bearish short-term trend.")
    else:
        notes.append("Mixed short-term trend.")

    if price > ema200:
        score += 2
        notes.append("Price is above the 200 EMA.")
    else:
        score -= 2
        notes.append("Price is below the 200 EMA.")

    if rsi < 30:
        score += 1
        notes.append("RSI suggests oversold conditions.")
    elif rsi > 70:
        score -= 1
        notes.append("RSI suggests overbought conditions.")
    else:
        notes.append("RSI is neutral.")

    if macd > macd_signal:
        score += 1
        notes.append("MACD is bullish.")
    else:
        score -= 1
        notes.append("MACD is bearish.")

    return {
        "price": round(price, 5),
        "rsi": round(rsi, 2),
        "ema20": round(ema20, 5),
        "ema50": round(ema50, 5),
        "ema200": round(ema200, 5),
        "macd": round(macd, 5),
        "macd_signal": round(macd_signal, 5),
        "score": score,
        "notes": notes
    }


def stock_fundamental_analysis(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info

    score = 0
    notes = []

    pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    profit_margin = info.get("profitMargins")
    revenue_growth = info.get("revenueGrowth")
    debt_to_equity = info.get("debtToEquity")
    recommendation = info.get("recommendationKey")

    if revenue_growth is not None:
        if revenue_growth > 0.10:
            score += 2
            notes.append("Revenue growth is strong.")
        elif revenue_growth > 0:
            score += 1
            notes.append("Revenue growth is positive.")
        else:
            score -= 1
            notes.append("Revenue growth is weak or negative.")

    if profit_margin is not None:
        if profit_margin > 0.15:
            score += 2
            notes.append("Profit margins are strong.")
        elif profit_margin > 0:
            score += 1
            notes.append("Company is profitable.")
        else:
            score -= 2
            notes.append("Company has negative profit margins.")

    if pe is not None:
        if pe < 25:
            score += 1
            notes.append("P/E valuation appears reasonable.")
        elif pe > 60:
            score -= 1
            notes.append("P/E valuation appears expensive.")

    if debt_to_equity is not None:
        if debt_to_equity < 100:
            score += 1
            notes.append("Debt level appears manageable.")
        else:
            score -= 1
            notes.append("Debt level appears elevated.")

    return {
        "company": info.get("longName", symbol),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap"),
        "pe": pe,
        "forward_pe": forward_pe,
        "profit_margin": profit_margin,
        "revenue_growth": revenue_growth,
        "debt_to_equity": debt_to_equity,
        "recommendation": recommendation,
        "score": score,
        "notes": notes
    }


def forex_qual_analysis(symbol):
    return {
        "macro_notes": [
            "Forex does not have company fundamentals like revenue, earnings, or P/E ratio.",
            "Qualitative forex analysis should focus on interest rates, inflation, central banks, risk sentiment, and economic strength.",
            "Use this technical scan as a directional filter, not as a complete macro forecast."
        ],
        "score": 0
    }


def ai_analysis(original_symbol, asset_type, technical, qualitative):
    prompt = f"""
You are a professional market analyst.

Analyze this asset using both quantitative and qualitative logic.

Symbol: {original_symbol}
Asset Type: {asset_type}

Technical Data:
{technical}

Qualitative/Fundamental/Macro Data:
{qualitative}

Return this format:

1. Overall Market Summary
2. Bullish Case
3. Bearish Case
4. Quant Signal
5. Qualitative Signal
6. Risk Level
7. Trade/Investment Rating
8. What To Watch Next

Do not promise profits. Do not give guaranteed financial advice.
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )

    return response.output_text


def final_rating(score):
    if score >= 6:
        return "Strong Bullish Watch"
    elif score >= 3:
        return "Bullish Watch"
    elif score >= 0:
        return "Neutral"
    elif score >= -3:
        return "Bearish Watch"
    else:
        return "Avoid / Strong Bearish"


def scan_symbol(user_symbol):
    clean_symbol, asset_type = clean_ticker(user_symbol)

    data = get_market_data(clean_symbol, asset_type)

    if data.empty:
        print(f"No data found for {user_symbol}")
        return

    technical = technical_analysis(data)

    if asset_type == "stock":
        qualitative = stock_fundamental_analysis(clean_symbol)
        total_score = technical["score"] + qualitative["score"]
    else:
        qualitative = forex_qual_analysis(clean_symbol)
        total_score = technical["score"]

    rating = final_rating(total_score)

    print("\n" + "=" * 80)
    print(f"SCAN RESULT: {user_symbol.upper()}")
    print(f"Asset Type: {asset_type.upper()}")
    print("=" * 80)

    print("\n--- Technical Analysis ---")
    print(f"Price: {technical['price']}")
    print(f"RSI: {technical['rsi']}")
    print(f"EMA20: {technical['ema20']}")
    print(f"EMA50: {technical['ema50']}")
    print(f"EMA200: {technical['ema200']}")
    print(f"MACD: {technical['macd']}")
    print(f"MACD Signal: {technical['macd_signal']}")
    print(f"Technical Score: {technical['score']}")

    for note in technical["notes"]:
        print(f"- {note}")

    print("\n--- Qualitative Analysis ---")

    if asset_type == "stock":
        print(f"Company: {qualitative['company']}")
        print(f"Sector: {qualitative['sector']}")
        print(f"Industry: {qualitative['industry']}")
        print(f"P/E: {qualitative['pe']}")
        print(f"Forward P/E: {qualitative['forward_pe']}")
        print(f"Profit Margin: {qualitative['profit_margin']}")
        print(f"Revenue Growth: {qualitative['revenue_growth']}")
        print(f"Debt to Equity: {qualitative['debt_to_equity']}")
        print(f"Analyst Recommendation: {qualitative['recommendation']}")
        print(f"Fundamental Score: {qualitative['score']}")

        for note in qualitative["notes"]:
            print(f"- {note}")
    else:
        for note in qualitative["macro_notes"]:
            print(f"- {note}")

    print("\n--- Final Scanner Rating ---")
    print(f"Total Score: {total_score}")
    print(f"Rating: {rating}")

    print("\n--- AI Market Analysis ---")
    try:
        ai_summary = ai_analysis(user_symbol.upper(), asset_type, technical, qualitative)
        print(ai_summary)
    except Exception as e:
        print("AI analysis failed. Check your OpenAI API key.")
        print(e)

    print("=" * 80)


def main():
    st.set_page_config(page_title="AI Stock & Forex Scanner", layout="wide")

    st.title("AI Stock & Forex Quant + Qual Scanner")

symbols = st.text_input(
    "Enter stocks or forex pairs separated by commas",
    value="AAPL, NVDA, EURUSD",
    key="main_symbol_input"
)

    run_scan = st.button("Run Scanner", key="run_scanner_button")

    if run_scan:
        symbol_list = symbols.split(",")

        for symbol in symbol_list:
            symbol = symbol.strip()

            if symbol:
                st.divider()
                st.subheader(f"Scan Result: {symbol.upper()}")

                clean_symbol, asset_type = clean_ticker(symbol)
                data = get_market_data(clean_symbol, asset_type)

                if data.empty:
                    st.error(f"No data found for {symbol}")
                    continue

                technical = technical_analysis(data)

                if asset_type == "stock":
                    qualitative = stock_fundamental_analysis(clean_symbol)
                    total_score = technical["score"] + qualitative["score"]
                else:
                    qualitative = forex_qual_analysis(clean_symbol)
                    total_score = technical["score"]

                rating = final_rating(total_score)

                st.write(f"**Asset Type:** {asset_type.upper()}")
                st.write(f"**Price:** {technical['price']}")
                st.write(f"**RSI:** {technical['rsi']}")
                st.write(f"**EMA20:** {technical['ema20']}")
                st.write(f"**EMA50:** {technical['ema50']}")
                st.write(f"**EMA200:** {technical['ema200']}")
                st.write(f"**Technical Score:** {technical['score']}")
                st.write(f"**Final Rating:** {rating}")

                st.write("### Technical Notes")
                for note in technical["notes"]:
                    st.write(f"- {note}")

                st.write("### Qualitative Analysis")
                if asset_type == "stock":
                    st.write(f"**Company:** {qualitative['company']}")
                    st.write(f"**Sector:** {qualitative['sector']}")
                    st.write(f"**Industry:** {qualitative['industry']}")
                    st.write(f"**P/E:** {qualitative['pe']}")
                    st.write(f"**Revenue Growth:** {qualitative['revenue_growth']}")
                    st.write(f"**Profit Margin:** {qualitative['profit_margin']}")
                    st.write(f"**Debt to Equity:** {qualitative['debt_to_equity']}")
                else:
                    for note in qualitative["macro_notes"]:
                        st.write(f"- {note}")

                st.write("### AI Market Analysis")
                with st.spinner("Generating AI analysis..."):
                    try:
                        ai_summary = ai_analysis(symbol.upper(), asset_type, technical, qualitative)
                        st.write(ai_summary)
                    except Exception as e:
                        st.error("AI analysis failed. Check your OpenAI API key.")
                        st.write(e)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()