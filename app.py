import streamlit as st
import os
import pandas as pd
import plotly.express as px
import yfinance as yf
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Talk to Your Portfolio", layout="wide")

# Load Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
if "NEON_DB_URL" in st.secrets:
    os.environ["NEON_DB_URL"] = st.secrets["NEON_DB_URL"]

# Initialize Models
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# --- 2. DATABASE & LIVE DATA ENGINE ---
try:
    # Database Connection
    db_url = os.environ["NEON_DB_URL"].replace("postgres://", "postgresql://")
    engine = create_engine(db_url)
    db = SQLDatabase(engine)
    sql_agent = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)
    db_status = "✅ Connected"
except Exception as e:
    db_status = f"❌ Error: {e}"
    sql_agent = None
    engine = None

# Connect to News
try:
    vector_store = PineconeVectorStore(index_name="market-news", embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
except:
    retriever = None

# Helper: Get Real-Time Portfolio Value
def get_portfolio_df():
    if not engine:
        return pd.DataFrame()
    
    # 1. Get Quantities from DB
    df = pd.read_sql("SELECT ticker, company_name, quantity, sector FROM holdings", engine)
    
    if df.empty:
        return df

    # 2. Get Live Prices from Yahoo Finance
    current_prices = []
    total_values = []
    
    for index, row in df.iterrows():
        try:
            ticker_data = yf.Ticker(row['ticker'])
            # Fast fetch of current price
            price = ticker_data.fast_info['last_price']
        except:
            price = 0.0 # Fallback if API fails
            
        current_prices.append(price)
        total_values.append(price * row['quantity'])
    
    df['current_price'] = current_prices
    df['total_value'] = total_values
    return df

# --- 3. THE BRAIN (Now with Live Context) ---
def get_financial_advice(query, portfolio_df):
    # A. Create a text summary of the LIVE portfolio
    if not portfolio_df.empty:
        portfolio_context = portfolio_df.to_string(index=False)
    else:
        portfolio_context = "Portfolio is empty."

    # B. Get Market News
    try:
        docs = retriever.invoke(query)
        news_context = "\n".join([d.page_content for d in docs])
    except:
        news_context = "No news found."

    # C. Synthesize Answer
    final_prompt = f"""
    You are a financial advisor.
    
    REAL-TIME PORTFOLIO DATA (Use this for values):
    {portfolio_context}
    
    MARKET NEWS:
    {news_context}
    
    USER QUESTION: {query}
    
    Instructions:
    - Use the 'total_value' column for portfolio worth.
    - If the user asks about buying/selling, analyze their current holdings first.
    - Be concise and professional.
    """
    
    return llm.invoke(final_prompt).content

# --- 4. THE UI ---
st.title("💰 Talk to Your Portfolio (Pro)")

# Fetch Live Data
df_portfolio = get_portfolio_df()

# ROW 1: METRICS CARDS
col1, col2, col3 = st.columns(3)
if not df_portfolio.empty:
    total_net_worth = df_portfolio['total_value'].sum()
    top_stock = df_portfolio.loc[df_portfolio['total_value'].idxmax()]
    
    col1.metric("Total Net Worth", f"${total_net_worth:,.2f}")
    col2.metric("Top Holding", f"{top_stock['ticker']}", f"${top_stock['total_value']:,.2f}")
    col3.metric("Portfolio Count", f"{len(df_portfolio)} Stocks")
else:
    col1.metric("Total Net Worth", "$0.00")

# ROW 2: CHARTS
c1, c2 = st.columns([2, 1])
with c1:
    if not df_portfolio.empty:
        fig = px.bar(df_portfolio, x='ticker', y='total_value', color='sector', title="Holdings by Value (Live)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to chart.")

with c2:
    st.write("### 💬 Chat Advisor")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ex: Should I sell Apple?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing live data..."):
                response = get_financial_advice(prompt, df_portfolio)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- 5. SIDEBAR: TRADE SIMULATOR & TOOLS ---
with st.sidebar:
    st.header("⚡ Trade Simulator")
    
    with st.form("trade_form"):
        ticker_input = st.text_input("Ticker (e.g., NVDA)").upper()
        quantity_input = st.number_input("Shares", min_value=1, value=10)
        action = st.selectbox("Action", ["BUY", "SELL"])
        
        submitted = st.form_submit_button("Execute Trade")
        
        if submitted and engine:
            with st.spinner("Executing..."):
                try:
                    # Check if stock exists in DB
                    exists_query = text("SELECT quantity FROM holdings WHERE ticker = :t")
                    with engine.connect() as conn:
                        result = conn.execute(exists_query, {"t": ticker_input}).fetchone()
                        
                        if action == "BUY":
                            if result:
                                # Update existing
                                new_qty = result[0] + quantity_input
                                conn.execute(text("UPDATE holdings SET quantity = :q WHERE ticker = :t"), {"q": new_qty, "t": ticker_input})
                            else:
                                # Insert new (requires sector/name fetch, using dummies for simplicity)
                                conn.execute(text("INSERT INTO holdings (ticker, company_name, quantity, sector, region, value) VALUES (:t, :t, :q, 'Unknown', 'USA', 0)"), {"t": ticker_input, "q": quantity_input})
                            conn.commit()
                            st.success(f"Bought {quantity_input} {ticker_input}!")
                            
                        elif action == "SELL":
                            if not result:
                                st.error("You don't own this stock!")
                            elif result[0] < quantity_input:
                                st.error(f"Cannot sell {quantity_input}. You only have {result[0]}.")
                            else:
                                new_qty = result[0] - quantity_input
                                if new_qty == 0:
                                    conn.execute(text("DELETE FROM holdings WHERE ticker = :t"), {"t": ticker_input})
                                else:
                                    conn.execute(text("UPDATE holdings SET quantity = :q WHERE ticker = :t"), {"q": new_qty, "t": ticker_input})
                                conn.commit()
                                st.success(f"Sold {quantity_input} {ticker_input}!")
                                
                except Exception as e:
                    st.error(f"Trade Failed: {e}")

    st.divider()
    st.subheader("⚙️ Admin")
    if st.button("Reset DB"):
        # (Keep your existing reset logic here if needed, omitted for brevity but recommended to keep)
        pass