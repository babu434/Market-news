import asyncio
import yfinance as yf
import pandas_ta as ta
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
import os
import streamlit as st

# --- CONFIGURATION ---
CONCURRENT_LIMIT = 5
search = DuckDuckGoSearchRun()

# ==========================================
# 🛡️ THE CACHING LAYER (Simulating Redis)
# ==========================================

@st.cache_data(ttl=300, show_spinner=False) # TTL: 5 Minutes
def fetch_price_history(ticker):
    return yf.Ticker(ticker).history(period="1y")

@st.cache_data(ttl=86400, show_spinner=False) # TTL: 24 Hours
def fetch_fundamentals(ticker):
    return yf.Ticker(ticker).info

@st.cache_data(ttl=14400, show_spinner=False) # TTL: 4 Hours
def fetch_search_results(query):
    return search.invoke(query)

@st.cache_data(ttl=300, show_spinner=False) # TTL: 5 Minutes
def generate_llm_response(prompt_str, inputs, _llm):
    # _llm is ignored by the hasher to prevent Streamlit caching errors
    prompt = ChatPromptTemplate.from_template(prompt_str)
    chain = prompt | _llm
    return chain.invoke(inputs).content

# ==========================================
# 🤖 THE SPECIALIZED AGENTS
# ==========================================

async def get_technical_data(ticker):
    try:
        # 1. Fetch from Cache
        df = await asyncio.to_thread(fetch_price_history, ticker)
        if df is None or df.empty: return None

        # 2. Process Math (Deterministic)
        df.ta.rsi(length=14, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        
        latest = df.iloc[-1]
        score = 0
        signals = []
        
        rsi = latest['RSI_14']
        if rsi < 30:
            score += 1
            signals.append("Oversold (RSI < 30) - Bullish")
        elif rsi > 70:
            score -= 1
            signals.append("Overbought (RSI > 70) - Bearish")
            
        if latest['EMA_50'] > latest['EMA_200']:
            score += 1
            signals.append("Golden Cross (Long-term Uptrend)")
        else:
            score -= 1
            signals.append("Death Cross (Long-term Downtrend)")

        return {
            "signal": "BUY" if score > 0 else "SELL" if score < 0 else "NEUTRAL",
            "rsi": round(rsi, 2),
            "close_prices": df['Close'].tolist(),
            "dates": df.index.strftime('%Y-%m-%d').tolist(),
            "details": signals
        }
    except Exception as e:
        print(f"❌ TECH ERROR ({ticker}): {e}")
        return None

async def get_fundamental_data(ticker, llm):
    try:
        info = await asyncio.to_thread(fetch_fundamentals, ticker)
        metrics = {
            "PE Ratio": info.get('trailingPE', 'N/A'),
            "Forward PE": info.get('forwardPE', 'N/A'),
            "PEG Ratio": info.get('pegRatio', 'N/A'),
            "Price to Book": info.get('priceToBook', 'N/A'),
            "Debt to Equity": info.get('debtToEquity', 'N/A')
        }

        prompt_str = """
        You are a Value Investor. Analyze these metrics for {ticker}: {metrics}
        Is this stock UNDERVALUED, OVERVALUED, or FAIR_VALUE?
        Give a 1-sentence reason focusing on PEG or PE.
        """
        response = await asyncio.to_thread(
            generate_llm_response, prompt_str, {"ticker": ticker, "metrics": str(metrics)}, llm
        )
        return {"verdict": response, "metrics": metrics}
    except Exception as e:
        return {"verdict": "Neutral (Data Missing)", "metrics": {}}

async def get_sentiment_data(ticker, llm):
    try:
        query = f"{ticker} stock news financial"
        news_text = await asyncio.to_thread(fetch_search_results, query)
        
        if not news_text or "No results" in news_text:
            return "NEUTRAL: No significant news found."

        prompt_str = """
        Analyze the following news for {ticker}.
        Output exactly one word: BULLISH, BEARISH, or NEUTRAL.
        Then add a dash and a 1-sentence summary.
        NEWS: {news}
        """
        response = await asyncio.to_thread(
            generate_llm_response, prompt_str, {"ticker": ticker, "news": news_text}, llm
        )
        return response
    except Exception as e:
        return f"NEUTRAL: Error ({str(e)})"

async def get_earnings_data(ticker, llm):
    try:
        query = f"{ticker} earnings call transcript key takeaways guidance management commentary"
        results = await asyncio.to_thread(fetch_search_results, query)
        
        if not results or "No results" in results:
            return {"status": "Neutral", "analysis": "No recent earnings guidance found."}

        prompt_str = """
        You are a Senior Analyst listening to an Earnings Call.
        Read these search snippets about {ticker}: {text}
        Extract:
        1. Management Tone (Confident/Cautious?)
        2. Future Guidance (Raising/Lowering outlook?)
        Output: "Tone: [Tone]. Summary: [1 sentence on Guidance]"
        """
        response = await asyncio.to_thread(
            generate_llm_response, prompt_str, {"ticker": ticker, "text": results}, llm
        )
        return {"status": "Success", "analysis": response}
    except Exception as e:
        return {"status": "Error", "analysis": "Data unavailable."}

# ==========================================
# 👑 THE CHAIRMAN AGENT
# ==========================================

async def get_verdict(ticker, tech, fund, sent, earn, llm):
    try:
        prompt_str = """
        You are a Hedge Fund Portfolio Manager.
        Ticker: {ticker}
        
        1. Technicals: {tech_sig} (RSI: {rsi})
        2. Fundamentals: {fund_verdict}
        3. Sentiment: {sent}
        4. Management Commentary: {earn_analysis}
        
        Synthesize these 4 signals.
        - IGNORE technicals if Management Guidance is disastrous.
        - BUY if Fundamentals are strong AND Management is confident, even if technicals are weak.
        
        Final Verdict (BUY/SELL/HOLD):
        Reason (1 sentence):
        """
        inputs = {
            "ticker": ticker, "tech_sig": tech['signal'], "rsi": tech['rsi'], 
            "fund_verdict": fund['verdict'], "sent": sent, "earn_analysis": earn['analysis']
        }
        response = await asyncio.to_thread(generate_llm_response, prompt_str, inputs, llm)
        return response
    except Exception as e:
        return f"HOLD: Analysis failed ({str(e)})"

# ==========================================
# ⚙️ ORCHESTRATION & BATCH PROCESSING
# ==========================================

async def analyze_single_stock(ticker, semaphore, llm):
    async with semaphore:
        tech_task = asyncio.create_task(get_technical_data(ticker))
        fund_task = asyncio.create_task(get_fundamental_data(ticker, llm))
        sent_task = asyncio.create_task(get_sentiment_data(ticker, llm))
        earn_task = asyncio.create_task(get_earnings_data(ticker, llm))
        
        tech, fund, sent, earn = await asyncio.gather(tech_task, fund_task, sent_task, earn_task)
        
        if not tech: 
            raise ValueError(f"Technical data missing for {ticker}")

        verdict = await get_verdict(ticker, tech, fund, sent, earn, llm)
        
        return {
            "ticker": ticker, "tech": tech, "fund": fund, 
            "sent": sent, "earn": earn, "verdict": verdict
        }

async def run_portfolio_analysis(tickers):
    llm = ChatGroq(model_name="llama-3.1-8b-instant")
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    
    tasks = [analyze_single_stock(t, semaphore, llm) for t in tickers]
    
    # Graceful Error Handling: return_exceptions=True prevents the whole batch from crashing
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out the exceptions and return only valid data
    valid_results = []
    for r in results:
        if isinstance(r, Exception):
            print(f"⚠️ A stock failed processing: {r}")
        elif r is not None:
            valid_results.append(r)
            
    return valid_results