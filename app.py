import streamlit as st
import os
import pandas as pd
import plotly.express as px
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Talk to Your Portfolio", layout="wide")

# Load Secrets (Works for both Local and Cloud)
# If running locally, you must set these in .streamlit/secrets.toml
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
if "NEON_DB_URL" in st.secrets:
    os.environ["NEON_DB_URL"] = st.secrets["NEON_DB_URL"]

# Initialize AI Models
# We use Llama 3 8B because it is fast and has high rate limits for free tiers
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Connect to Databases
# Note: These might fail if the DB is not initialized yet (handled below)
try:
    db = SQLDatabase.from_uri(os.environ["NEON_DB_URL"])
    sql_agent = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)
except:
    db = None
    sql_agent = None

vector_store = PineconeVectorStore(index_name="market-news", embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --- 2. THE INTELLIGENCE LAYER ---
def get_financial_advice(query):
    # Step A: Get Portfolio Data
    if not sql_agent:
        return "⚠️ Error: Database not connected. Please use the Sidebar to Initialize DB."
    
    try:
        # We ask the SQL agent to summarize the holdings
        portfolio_context = sql_agent.invoke("Summarize my holdings including company names, quantity, and total value.")['output']
    except Exception as e:
        portfolio_context = "⚠️ Error reading database. The table might be missing. Go to Sidebar -> Reset Database."

    # Step B: Get Market News
    try:
        docs = retriever.invoke(query)
        news_context = "\n".join([d.page_content for d in docs])
    except:
        news_context = "No news found."

    # Step C: Synthesize Answer
    final_prompt = f"""
    You are a financial advisor.
    
    User Portfolio: {portfolio_context}
    Market News: {news_context}
    User Question: {query}
    
    Instructions:
    1. If the user has the stock in their portfolio, mention their exact share count and value.
    2. Explain how the news specifically affects their holdings.
    3. If the portfolio data is an error message, tell the user to fix it in the sidebar.
    """
    
    return llm.invoke(final_prompt).content

# --- 3. THE USER INTERFACE ---
st.title("💰 Talk to Your Portfolio")

# Top Row: Visualization
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Asset Allocation")
    try:
        # We use SQLAlchemy to fetch data for the chart
        engine = create_engine(os.environ["NEON_DB_URL"])
        df = pd.read_sql("SELECT company_name, value, sector FROM holdings", engine)
        
        if not df.empty:
            fig = px.pie(df, values='value', names='sector', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Portfolio is empty.")
    except Exception:
        st.warning("⚠️ Database not ready. Click 'Reset Database' in the sidebar.")

with col2:
    st.info("👋 Welcome! I can see your SQL portfolio and read live market news.")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask: 'Should I buy more Apple?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_financial_advice(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- 4. THE ADMIN SIDEBAR (INITIATION TOOLS) ---
with st.sidebar:
    st.header("⚙️ Admin Setup")
    st.write("Use these buttons to initialize your app for the first time.")

    # --- TOOL A: INITIALIZE DATABASE (NEON) ---
    st.divider()
    st.subheader("1. Initialize Database")
    if st.button("Create/Reset Portfolio DB"):
        with st.spinner("Creating SQL Tables..."):
            try:
                import psycopg2
                conn = psycopg2.connect(os.environ["NEON_DB_URL"])
                cur = conn.cursor()
                
                # 1. Drop old table
                cur.execute("DROP TABLE IF EXISTS holdings;")
                
                # 2. Create new table (The 'Initiation')
                cur.execute("""
                    CREATE TABLE holdings (
                        id SERIAL PRIMARY KEY,
                        ticker VARCHAR(10),
                        company_name VARCHAR(100),
                        quantity INT,
                        sector VARCHAR(50),
                        region VARCHAR(50),
                        value DECIMAL(10, 2)
                    );
                """)
                
                # 3. Add Dummy Data
                cur.execute("""
                    INSERT INTO holdings (ticker, company_name, quantity, sector, region, value) VALUES 
                    ('AAPL', 'Apple Inc.', 100, 'Technology', 'USA', 17500.00),
                    ('SHEL', 'Shell PLC', 500, 'Energy', 'Europe', 32000.00),
                    ('TSLA', 'Tesla Inc.', 50, 'Automotive', 'USA', 12000.00);
                """)
                
                conn.commit()
                conn.close()
                st.success("✅ Database Initialized! (Refresh Page)")
            except Exception as e:
                st.error(f"DB Init Failed: {e}")

    # --- TOOL B: INITIALIZE NEWS (PINECONE) ---
    st.divider()
    st.subheader("2. Initialize News")
    if st.button("Fetch Live News"):
        with st.spinner("Scraping Yahoo Finance..."):
            try:
                import time
                import yfinance as yf
                from pinecone import Pinecone, ServerlessSpec
                
                # 1. Setup Pinecone
                pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
                INDEX_NAME = "market-news"
                
                # Delete old index if exists (Clean Slate)
                if INDEX_NAME in [i.name for i in pc.list_indexes()]:
                    pc.delete_index(INDEX_NAME)
                    time.sleep(5)
                
                # Create new index
                pc.create_index(
                    name=INDEX_NAME, dimension=768, metric="cosine", 
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(10)
                
                # 2. Get News
                tickers = ["AAPL", "SHEL", "TSLA"]
                news_docs = []
                for t in tickers:
                    stock = yf.Ticker(t)
                    for news in stock.news[:2]: # Top 2 articles per stock
                        news_docs.append(f"News for {t}: {news['title']}")
                
                # 3. Upload
                PineconeVectorStore.from_texts(news_docs, embeddings, index_name=INDEX_NAME)
                st.success("✅ News Initialized!")
            except Exception as e:
                st.error(f"News Init Failed: {e}")