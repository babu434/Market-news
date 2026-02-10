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

# --- 2. THE CRITICAL FIX (SHARED ENGINE) ---
# We create ONE connection engine and share it.
# This fixes the issue where the Chart sees data but the Agent doesn't.
try:
    # Ensure URL uses the correct driver
    db_url = os.environ["NEON_DB_URL"].replace("postgres://", "postgresql://")
    
    # Create the Engine
    engine = create_engine(db_url)
    
    # Give the Engine to the AI Agent
    db = SQLDatabase(engine)
    sql_agent = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)
    
    db_status = "✅ Connected"
except Exception as e:
    db_status = f"❌ Error: {e}"
    sql_agent = None

# Connect to News (Vector DB)
try:
    vector_store = PineconeVectorStore(index_name="market-news", embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
except:
    retriever = None

# --- 3. THE BRAIN ---
def get_financial_advice(query):
    # A. Get Portfolio Data (Using the Shared Engine)
    try:
        if sql_agent:
            # We explicitly ask for the table 'holdings' to force the agent to look there
            portfolio_context = sql_agent.invoke(
                f"Query the 'holdings' table. {query}. Return the answer as a summary."
            )['output']
        else:
            portfolio_context = "Database Connection Failed."
    except Exception as e:
        portfolio_context = f"Error querying database: {str(e)}"

    # B. Get Market News
    try:
        docs = retriever.invoke(query)
        news_context = "\n".join([d.page_content for d in docs])
    except:
        news_context = "No news found."

    # C. Synthesize Answer
    final_prompt = f"""
    You are a financial advisor.
    
    User Portfolio Data: {portfolio_context}
    Market News: {news_context}
    User Question: {query}
    
    Instructions:
    - If the portfolio data shows share counts, stated them clearly.
    - If the portfolio data is an error, tell the user.
    - Answer the question using both data sources.
    """
    
    return llm.invoke(final_prompt).content

# --- 4. THE UI ---
st.title("💰 Talk to Your Portfolio")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Asset Allocation")
    # Use the SAME engine for the chart
    try:
        if 'engine' in locals():
            df = pd.read_sql("SELECT company_name, value, sector FROM holdings", engine)
            if not df.empty:
                fig = px.pie(df, values='value', names='sector', hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Database connected but empty.")
    except Exception as e:
        st.error(f"Chart Error: {e}")

with col2:
    st.info(f"System Status: {db_status}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask: What shares do I own?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_financial_advice(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- 5. ADMIN TOOLS ---
with st.sidebar:
    st.header("⚙️ Admin Tools")
    
    if st.button("Re-Create Database"):
        with st.spinner("Resetting..."):
            try:
                import psycopg2
                # Direct connection for DDL commands
                conn = psycopg2.connect(db_url)
                cur = conn.cursor()
                cur.execute("DROP TABLE IF EXISTS holdings;")
                cur.execute("""
                    CREATE TABLE holdings (
                        ticker VARCHAR(10),
                        company_name VARCHAR(100),
                        quantity INT,
                        sector VARCHAR(50),
                        region VARCHAR(50),
                        value DECIMAL(10, 2)
                    );
                """)
                cur.execute("""
                    INSERT INTO holdings VALUES 
                    ('AAPL', 'Apple Inc.', 100, 'Technology', 'USA', 17500.00),
                    ('SHEL', 'Shell PLC', 500, 'Energy', 'Europe', 32000.00),
                    ('TSLA', 'Tesla Inc.', 50, 'Automotive', 'USA', 12000.00);
                """)
                conn.commit()
                conn.close()
                st.success("✅ Database Reset!")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("Update News"):
        with st.spinner("Updating..."):
            try:
                import yfinance as yf
                from pinecone import Pinecone, ServerlessSpec
                pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
                if "market-news" in [i.name for i in pc.list_indexes()]:
                    pc.delete_index("market-news")
                pc.create_index("market-news", dimension=768, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
                
                news_docs = []
                for t in ["AAPL", "SHEL", "TSLA"]:
                    stock = yf.Ticker(t)
                    for n in stock.news[:2]:
                        news_docs.append(f"News for {t}: {n['title']}")
                
                PineconeVectorStore.from_texts(news_docs, embeddings, index_name="market-news")
                st.success("✅ News Updated!")
            except Exception as e:
                st.error(f"Error: {e}")