import asyncio
import yfinance as yf
import pandas_ta as ta
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
import os

# --- CONFIGURATION ---
CONCURRENT_LIMIT = 5  # Max 5 stocks at once to avoid Rate Limits
search = DuckDuckGoSearchRun()

# --- AGENT 1: TECHNICAL ANALYST (Math + Data) ---
async def get_technical_data(ticker):
    """
    Fetches 1y history, calculates RSI/EMA, and prepares chart data.
    """
    try:
        # Run yfinance in a separate thread to not block the event loop
        df = await asyncio.to_thread(yf.Ticker(ticker).history, period="1y")
        
        if df.empty: return None

        # Calculate Indicators
        df.ta.rsi(length=14, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        
        latest = df.iloc[-1]
        
        # Logic
        signals = []
        score = 0
        
        # RSI Logic
        rsi = latest['RSI_14']
        if rsi < 30:
            signals.append("Oversold (Bullish)")
            score += 1
        elif rsi > 70:
            signals.append("Overbought (Bearish)")
            score -= 1
            
        # Trend Logic
        if latest['EMA_50'] > latest['EMA_200']:
            signals.append("Uptrend (Golden Cross)")
            score += 1
        else:
            signals.append("Downtrend (Death Cross)")
            score -= 1

        return {
            "status": "success",
            "signal": "BUY" if score > 0 else "SELL" if score < 0 else "NEUTRAL",
            "rsi": round(rsi, 2),
            "close_prices": df['Close'].tolist(), # For the Line Graph
            "dates": df.index.strftime('%Y-%m-%d').tolist(),
            "details": signals
        }
    except Exception as e:
        return {"status": "error", "msg": str(e)}

# --- AGENT 2: SENTIMENT ANALYST (News + AI) ---
async def get_sentiment_data(ticker, llm):
    """
    Searches news and uses LLM to judge sentiment.
    """
    try:
        # 1. Search (Async wrapper)
        query = f"{ticker} stock news financial"
        news_text = await asyncio.to_thread(search.invoke, query)
        
        # 2. Guardrail: Check for empty news
        if not news_text or "No results" in news_text:
            return "NEUTRAL: No significant news found."

        # 3. AI Analysis
        prompt = ChatPromptTemplate.from_template("""
        Analyze the following news for {ticker}.
        Output exactly one word: BULLISH, BEARISH, or NEUTRAL.
        Then add a dash and a 1-sentence summary.
        Example: BULLISH - Earnings beat expectations.
        
        NEWS: {news}
        """)
        chain = prompt | llm
        response = await chain.ainvoke({"ticker": ticker, "news": news_text})
        return response.content
    except:
        return "NEUTRAL: Error fetching news."

# --- AGENT 3: THE CHAIRMAN (Synthesis) ---
async def get_verdict(ticker, tech, sent, llm):
    try:
        prompt = ChatPromptTemplate.from_template("""
        You are a Portfolio Manager.
        Ticker: {ticker}
        Technical Signal: {tech_sig} (RSI: {rsi})
        Sentiment: {sent}
        
        Decide: BUY, SELL, or HOLD.
        Give a 1-sentence reason.
        """)
        chain = prompt | llm
        response = await chain.ainvoke({
            "ticker": ticker, 
            "tech_sig": tech['signal'], 
            "rsi": tech['rsi'], 
            "sent": sent
        })
        return response.content
    except:
        return "HOLD: Analysis failed."

# --- THE BATCH PROCESSOR (The Engine) ---
async def analyze_single_stock(ticker, semaphore, llm):
    async with semaphore:  # Limits concurrency
        # Run Agents in Parallel for this ONE stock
        tech_task = asyncio.create_task(get_technical_data(ticker))
        sent_task = asyncio.create_task(get_sentiment_data(ticker, llm))
        
        tech, sent = await asyncio.gather(tech_task, sent_task)
        
        if not tech or tech['status'] == 'error':
            return None

        # Run Chairman after agents finish
        verdict = await get_verdict(ticker, tech, sent, llm)
        
        return {
            "ticker": ticker,
            "tech": tech,
            "sent": sent,
            "verdict": verdict
        }

async def run_portfolio_analysis(tickers):
    """
    Main entry point to analyze a list of tickers.
    """
    llm = ChatGroq(model_name="llama-3.1-8b-instant")
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    
    tasks = [analyze_single_stock(t, semaphore, llm) for t in tickers]
    results = await asyncio.gather(*tasks)
    
    # Filter out failed stocks
    return [r for r in results if r is not None]