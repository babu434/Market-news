import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import asyncio
import os
from engine import run_portfolio_analysis

st.set_page_config(page_title="Market-News Pro Ultra", layout="wide")

st.title("⚡ Market-News Pro: 4-Agent AI System")
st.markdown("""
**An Enterprise-Grade Agentic System** featuring:
* Multi-Agent Orchestration (Technicals, Fundamentals, Sentiment, Earnings)
* Asynchronous Batch Processing
* Multi-Tiered Cache-Aside Pattern (Granular TTLs)
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if api_key: os.environ["GROQ_API_KEY"] = api_key
    
    st.divider()
    st.subheader("Your Portfolio")
    default_tickers = "AAPL, MSFT, TSLA, NVDA, INTC"
    ticker_input = st.text_area("Enter Tickers:", default_tickers, height=100)
    
    run_btn = st.button("🚀 Run AI Analysis", type="primary", use_container_width=True)

# --- CHART HELPER ---
def create_sparkline(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['dates'], y=data['close_prices'], 
        mode='lines', line=dict(color='#00CC96', width=2),
        fill='tozeroy', fillcolor='rgba(0, 204, 150, 0.1)'
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=50,
        xaxis_visible=False, yaxis_visible=False, showlegend=False
    )
    return fig

# --- MAIN EXECUTION ---
if run_btn:
    if not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ Please enter a Groq API Key.")
    else:
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        
        with st.status(f"🤖 Processing {len(tickers)} stocks asynchronously...", expanded=True) as status:
            results = asyncio.run(run_portfolio_analysis(tickers))
            status.update(label=f"✅ Successfully analyzed {len(results)} stocks!", state="complete", expanded=False)

        st.divider()
        cols = st.columns(2) 
        
        for i, r in enumerate(results):
            col = cols[i % 2]
            with col:
                with st.container(border=True):
                    # Header
                    raw_verdict = r['verdict'].upper()
                    color = "green" if "BUY" in raw_verdict else "red" if "SELL" in raw_verdict else "orange"
                    final_v = "BUY" if "BUY" in raw_verdict else "SELL" if "SELL" in raw_verdict else "HOLD"
                    
                    c1, c2 = st.columns([2, 1])
                    c1.markdown(f"### {r['ticker']}")
                    c2.markdown(f"<h3 style='text-align: right; color: {color};'>{final_v}</h3>", unsafe_allow_html=True)
                    
                    # Core Metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("RSI", r['tech']['rsi'])
                    m2.metric("PE Ratio", r['fund']['metrics'].get('PE Ratio', 'N/A'))
                    m3.metric("Sentiment", "Bullish" if "BULLISH" in r['sent'] else "Bearish" if "BEARISH" in r['sent'] else "Neutral")
                    m4.metric("Technicals", r['tech']['signal'])
                    
                    st.plotly_chart(create_sparkline(r['tech']), use_container_width=True)
                    st.info(f"**Chairman's Rationale:** {r['verdict']}")
                    
                    # Deep Dive Reports
                    with st.expander("🔍 View All 4 Agent Reports"):
                        st.markdown("**1️⃣ Technical Analyst:**")
                        for s in r['tech']['details']: st.caption(f"• {s}")
                        
                        st.markdown("**2️⃣ Fundamental Analyst:**")
                        st.caption(f"{r['fund']['verdict']}")
                        
                        st.markdown("**3️⃣ Sentiment Analyst:**")
                        st.caption(r['sent'])
                        
                        st.markdown("**4️⃣ Earnings & Management Analyst:**")
                        st.caption(r['earn']['analysis'])