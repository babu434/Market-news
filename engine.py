import asyncio
import yfinance as yf
import pandas_ta as ta
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
import os

# --- CONFIGURATION ---
# Semaphore limits concurrency to prevent API bans.
# Lower to 3 if you see rate limit errors.
CONCURRENT_LIMIT = 5
search = DuckDuckGoSearchRun()

# --- AGENT 1: TECHNICAL ANALYST (Math & Trend) ---
async def get_technical_data(ticker):
    """
    Determines trend using EMA crossovers and RSI.
    """
    try:
        # Run blocking yfinance call in a separate thread
        df = await asyncio.to_thread(yf.Ticker(ticker).history, period="1y")
        if df.empty: return None

        # Calculate Indicators
        df.ta.rsi(length=14, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        
        latest = df.iloc[-1]
        score = 0
        signals = []
        
        # RSI Logic
        rsi = latest['RSI_14']
        if rsi < 30:
            score += 1
            signals.append("Oversold (RSI < 30) - Bullish")
        elif rsi > 70:
            score -= 1
            signals.append("Overbought (RSI > 70) - Bearish")
            
        # Trend Logic (Golden/Death Cross)
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

# --- AGENT 2: FUNDAMENTAL ANALYST (Value & Health) ---
async def get_fundamental_data(ticker, llm):
    """
    Analyzes valuation metrics (PE, PEG, Debt).
    """
    try:
        stock = yf.Ticker(ticker)
        info = await asyncio.to_thread(lambda: stock.info)
        
        metrics = {
            "PE Ratio": info.get('trailingPE', 'N/A'),
            "Forward PE": info.get('forwardPE', 'N/A'),
            "PEG Ratio": info.get('pegRatio', 'N/A'),
            "Price to Book": info.get('priceToBook', 'N/A'),
            "Debt to Equity": info.get('debtToEquity', 'N/A'),
            "Profit Margin": info.get('profitMargins', 'N/A')
        }

        prompt = ChatPromptTemplate.from_template("""
        You are a Value Investor like Warren Buffett.
        Analyze these metrics for {ticker}: {metrics}
        
        Is this stock UNDERVALUED, OVERVALUED, or FAIR_VALUE?
        Give a 1-sentence reason focusing on PEG or PE.
        """)
        chain = prompt | llm
        response = await chain.ainvoke({"ticker": ticker, "metrics": str(metrics)})
        
        return {"verdict": response.content, "metrics": metrics}
    except Exception as e:
        return {"verdict": "Neutral (Data Missing)", "metrics": {}}

# --- AGENT 3: SENTIMENT ANALYST (News Cycles) ---
async def get_sentiment_data(ticker, llm):
    """
    Analyzes general market news for short-term sentiment.
    """
    try:
        query = f"{ticker} stock news financial"
        news_text = await asyncio.to_thread(search.invoke, query)
        
        if not news_text or "No results" in news_text:
            return "NEUTRAL: No significant news found."

        prompt = ChatPromptTemplate.from_template("""
        Analyze the following news for {ticker}.
        Output exactly one word: BULLISH, BEARISH, or NEUTRAL.
        Then add a dash and a 1-sentence summary.
        NEWS: {news}
        """)
        chain = prompt | llm
        response = await chain.ainvoke({"ticker": ticker, "news": news_text})
        return response.content
    except Exception as e:
        return f"NEUTRAL: Error ({str(e)})"

# --- AGENT 4: EARNINGS & MANAGEMENT ANALYST (Strategic Vision) ---
async def get_earnings_data(ticker, llm):
    """
    Analyzes Earnings Call transcripts and Management Guidance.
    """
    try:
        # Targeted search for "Guidance" and "Transcripts"
        query = f"{ticker} earnings call transcript key takeaways guidance management commentary"
        results = await asyncio.to_thread(search.invoke, query)
        
        if not results or "No results" in results:
            return {"status": "Neutral", "analysis": "No recent earnings guidance found."}

        prompt = ChatPromptTemplate.from_template("""
        You are a Senior Analyst listening to an Earnings Call.
        Read these search snippets about {ticker}: {text}
        
        Extract:
        1. Management Tone (Confident/Cautious?)
        2. Future Guidance (Raising/Lowering outlook?)
        
        Output: "Tone: [Tone]. Summary: [1 sentence on Guidance]"
        """)
        chain = prompt | llm
        response = await chain.ainvoke({"ticker": ticker, "text": results})
        return {"status": "Success", "analysis": response.content}
    except Exception as e:
        return {"status": "Error", "analysis": "Data unavailable."}

# --- AGENT 5: THE CHAIRMAN (Synthesis) ---
async def get_verdict(ticker, tech, fund, sent, earn, llm):
    """
    Synthesizes reports from all 4 agents to make a final decision.
    """
    try:
        prompt = ChatPromptTemplate.from_template("""
        You are a Hedge Fund Portfolio Manager.
        Ticker: {ticker}
        
        1. Technicals: {tech_sig} (RSI: {rsi})
        2. Fundamentals: {fund_verdict}
        3. Sentiment: {sent}
        4. Management Commentary: {earn_analysis}
        
        Synthesize these 4 signals.
        - IGNORE technicals if Management Guidance is disastrous.
        - BUY if Fundamentals are strong AND Management is confident, even if technicals are weak (Turnaround play).
        
        Final Verdict (BUY/SELL/HOLD):
        Reason (1 sentence):
        """)
        chain = prompt | llm
        response = await chain.ainvoke({
            "ticker": ticker, 
            "tech_sig": tech['signal'], 
            "rsi": tech['rsi'], 
            "fund_verdict": fund['verdict'],
            "sent": sent,
            "earn_analysis": earn['analysis']
        })
        return response.content
    except Exception as e:
        return f"HOLD: Analysis failed ({str(e)})"

# --- THE ENGINE (Batch Processor) ---
async def analyze_single_stock(ticker, semaphore, llm):
    async with semaphore:
        # 1. Fire all 4 Agents in Parallel
        tech_task = asyncio.create_task(get_technical_data(ticker))
        fund_task = asyncio.create_task(get_fundamental_data(ticker, llm))
        sent_task = asyncio.create_task(get_sentiment_data(ticker, llm))
        earn_task = asyncio.create_task(get_earnings_data(ticker, llm))
        
        # 2. Wait for all to return
        tech, fund, sent, earn = await asyncio.gather(tech_task, fund_task, sent_task, earn_task)
        
        if not tech: return None # Basic sanity check

        # 3. The Chairman Decides
        verdict = await get_verdict(ticker, tech, fund, sent, earn, llm)
        
        return {
            "ticker": ticker,
            "tech": tech,
            "fund": fund,
            "sent": sent,
            "earn": earn,
            "verdict": verdict
        }

async def run_portfolio_analysis(tickers):
    # Initialize the "Brain"
    llm = ChatGroq(model_name="llama-3.1-8b-instant")
    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    
    tasks = [analyze_single_stock(t, semaphore, llm) for t in tickers]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]