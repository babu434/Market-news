import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import asyncio
import os
# Make sure this matches your filename (engine.py)
from engine import run_portfolio_analysis

# --- PAGE CONFIG ---
st.set_page_config(page_title="Market-News Pro", layout="wide")

st.title("⚡ Market-News Pro: Agentic AI Analyst")
st.markdown("Analyze your portfolio using **3 Autonomous AI Agents** (Technical, Fundamental, Sentiment).")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Input (Secure)
    api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
    
    st.divider()
    
    # Portfolio Input
    st.subheader("Your Portfolio")
    default_tickers = "AAPL, NVDA, TSLA, AMD, MSFT, GOOGL"
    ticker_input = st.text_area("Enter Tickers (comma separated):", default_tickers, height=150)
    
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# --- HELPER: DRAW SPARKLINE ---
def create_sparkline(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['dates'], 
        y=data['close_prices'], 
        mode='lines', 
        line=dict(color='#00CC96', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 204, 150, 0.1)'
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=50,
        xaxis_visible=False, 
        yaxis_visible=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

# --- MAIN APP LOGIC ---
if run_btn:
    if not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ Please enter a Groq API Key in the sidebar.")
    else:
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        
        # 1. RUNNING THE AGENTS
        with st.status(f"🤖 Agents analyzing {len(tickers)} stocks...", expanded=True) as status:
            results = asyncio.run(run_portfolio_analysis(tickers))
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        # 2. DASHBOARD DISPLAY
        st.divider()
        
        # Grid Layout (2 Columns per row)
        cols = st.columns(2)
        
        for i, r in enumerate(results):
            col = cols[i % 2]
            
            with col:
                with st.container(border=True):
                    # --- HEADER: Ticker + Verdict ---
                    # Clean up the verdict string
                    raw_verdict = r['verdict'].upper()
                    if "BUY" in raw_verdict:
                        verdict_color = "green"
                        verdict_text = "BUY"
                    elif "SELL" in raw_verdict:
                        verdict_color = "red"
                        verdict_text = "SELL"
                    else:
                        verdict_color = "orange"
                        verdict_text = "HOLD"
                    
                    # Top Row: Ticker (Left) | Verdict (Right)
                    c1, c2 = st.columns([1, 1])
                    c1.markdown(f"### {r['ticker']}")
                    c2.markdown(f"<h3 style='text-align: right; color: {verdict_color};'>{verdict_text}</h3>", unsafe_allow_html=True)
                    
                    # --- METRICS ROW ---
                    m1, m2, m3 = st.columns(3)
                    m1.metric("RSI", f"{r['tech']['rsi']}")
                    m2.metric("Sentiment", "Bullish" if "BULLISH" in r['sent'] else "Bearish" if "BEARISH" in r['sent'] else "Neutral")
                    m3.metric("Price Trend", r['tech']['signal'])
                    
                    # --- CHART ---
                    st.plotly_chart(create_sparkline(r['tech']), use_container_width=True)
                    
                    # --- AI REASONING (The "Chairman's" Voice) ---
                    st.info(f"**AI Rationale:** {r['verdict']}")
                    
                    # --- EXPANDER FOR RAW DETAILS ---
                    with st.expander("View Agent Reports"):
                        st.markdown("**Technical Analyst:**")
                        for sig in r['tech']['details']:
                            st.caption(f"• {sig}")
                        
                        st.markdown("**Sentiment Analyst:**")
                        st.caption(r['sent'])