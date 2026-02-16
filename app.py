import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import asyncio
import os
from engine import run_portfolio_analysis

# --- PAGE CONFIG ---
st.set_page_config(page_title="Market-News Pro", layout="wide")

st.title("⚡ Market-News Pro: Agentic Portfolio Tracker")
st.markdown("Analyze 100+ stocks in seconds using **Async AI Agents**.")

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("🔑 API Keys")
    api_key = st.text_input("Groq API Key", type="password")
    if api_key: os.environ["GROQ_API_KEY"] = api_key
    
    st.divider()
    
    st.header("📂 Portfolio")
    default_tickers = "AAPL, TSLA, NVDA, MSFT, AMD, AMZN, GOOGL, META"
    ticker_input = st.text_area("Enter Tickers (comma separated)", default_tickers, height=150)
    
    run_btn = st.button("🚀 Run Analysis", type="primary")

# --- MAIN LOGIC ---
if run_btn:
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please enter a Groq API Key.")
    else:
        # 1. Parse Input
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        
        # 2. Run Engine (Async)
        with st.status(f"🤖 Agents Analyzing {len(tickers)} Stocks...", expanded=True) as status:
            # We must create a new loop for Streamlit
            results = asyncio.run(run_portfolio_analysis(tickers))
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        # 3. Display Results (Grid Layout)
        st.divider()
        
        # Convert results to DataFrame for a Summary Table
        summary_data = []
        for r in results:
            summary_data.append({
                "Ticker": r['ticker'],
                "Verdict": r['verdict'].split(":")[0], # Extract "BUY"
                "RSI": r['tech']['rsi'],
                "Sentiment": "Bullish" if "BULLISH" in r['sent'] else "Bearish" if "BEARISH" in r['sent'] else "Neutral"
            })
        
        st.subheader("📊 Portfolio Overview")
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        st.divider()
        st.subheader("🔍 Deep Dive & Charts")

        # Create Rows of 2 columns
        cols = st.columns(2)
        
        for i, r in enumerate(results):
            col = cols[i % 2] # Alternating columns
            
            with col:
                with st.container(border=True):
                    # Header: Ticker + Verdict
                    c1, c2 = st.columns([1, 3])
                    c1.markdown(f"## {r['ticker']}")
                    
                    verdict_color = "green" if "BUY" in r['verdict'] else "red" if "SELL" in r['verdict'] else "orange"
                    c2.markdown(f":{verdict_color}[**{r['verdict']}**]")
                    
                    # Line Chart (Plotly)
                    prices = r['tech']['close_prices']
                    dates = r['tech']['dates']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates, y=prices, 
                        mode='lines', 
                        line=dict(color='#00CC96', width=2),
                        fill='tozeroy'
                    ))
                    fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=150,
                        xaxis_visible=False, 
                        yaxis_visible=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Agent Details
                    st.info(f"**News:** {r['sent']}")
                    
                    # Expandable Technicals
                    with st.expander("See Technical Details"):
                        st.write(f"**RSI:** {r['tech']['rsi']}")
                        for sig in r['tech']['details']:
                            st.write(f"- {sig}")