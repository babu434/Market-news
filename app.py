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
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# Database Connections
db = SQLDatabase.from_uri(os.environ["NEON_DB_URL"])
sql_agent = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)

vector_store = PineconeVectorStore(index_name="market-news", embedding=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --- 2. THE BRAIN (LOGIC) ---
def get_financial_advice(query):
    # A. Always get Portfolio Context (The "Memory")
    try:
        portfolio_context = sql_agent.invoke("Summarize my holdings including company names, quantity, and total value.")['output']
    except:
        portfolio_context = "Error retrieving portfolio."

    # B. Always get News Context (The "Eyes")
    try:
        docs = retriever.invoke(query)
        news_context = "\n".join([d.page_content for d in docs])
    except:
        news_context = "No news found."

    # C. Synthesize Answer
    final_prompt = f"""
    You are a professional financial advisor.
    
    Data Source 1 (Portfolio): {portfolio_context}
    Data Source 2 (Market News): {news_context}
    User Question: {query}
    
    Instructions:
    - Synthesize the portfolio data with the market news.
    - Be specific: Mention exact share counts and values from the portfolio.
    - If the news explains a price movement for a holding, mention it.
    """
    
    return llm.invoke(final_prompt).content

# --- 3. THE UI (VISUALS & CHAT) ---
st.title("💰 Talk to Your Portfolio (Ultimate Edition)")

# ROW 1: Portfolio Visualization (Real-time SQL)
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Your Allocation")
    # We run a raw SQL query to get data for the chart
    try:
        # Note: We use pandas to read directly from the SQL connection
        # This requires the 'psycopg2' driver we installed
        from sqlalchemy import create_engine
        engine = create_engine(os.environ["NEON_DB_URL"])
        df = pd.read_sql("SELECT company_name, value, sector FROM holdings", engine)
        
        if not df.empty:
            fig = px.pie(df, values='value', names='sector', hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No holdings found in database.")
    except Exception as e:
        st.error(f"Could not load chart: {e}")

with col2:
    st.info("💡 **Try asking:** 'How does today's news affect my Apple stock?' or 'What is my total exposure to Energy?'")

# ROW 2: Chat Interface with History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about your holdings..."):
    # 1. Add User Message to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing portfolio & news..."):
            response = get_financial_advice(prompt)
            st.markdown(response)
    
    # 3. Add AI Response to History
    st.session_state.messages.append({"role": "assistant", "content": response})

# --- 4. ADMIN SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Admin Tools")

    # TOOL 1: REAL NEWS
    if st.button("Fetch LIVE News (Yahoo)"):
        with st.spinner("Fetching news..."):
            try:
                import yfinance as yf
                from pinecone import Pinecone, ServerlessSpec
                from langchain_pinecone import PineconeVectorStore
                
                # Connect & Reset
                pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
                INDEX_NAME = "market-news"
                
                if INDEX_NAME in [i.name for i in pc.list_indexes()]:
                    pc.delete_index(INDEX_NAME)
                
                pc.create_index(
                    name=INDEX_NAME, dimension=768, metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                
                # Fetch & Upload
                news_docs = []
                for ticker in ["AAPL", "SHEL"]:
                    stock = yf.Ticker(ticker)
                    for item in stock.news[:3]:
                        news_docs.append(f"News for {ticker}: {item['title']}")
                
                PineconeVectorStore.from_texts(news_docs, embeddings, index_name=INDEX_NAME)
                st.success("✅ News Updated!")
            except Exception as e:
                st.error(f"News Error: {e}")

    # TOOL 2: DATABASE REPAIR (Must be outside the first button!)
    st.divider()
    st.header("⚠️ Database Repair")
    
    if st.button("Reset Database (Fix Missing Column)"):
        with st.spinner("Resetting table..."):
            try:
                import psycopg2
                conn = psycopg2.connect(os.environ["NEON_DB_URL"])
                cur = conn.cursor()
                
                # Nuke and Rebuild
                cur.execute("DROP TABLE IF EXISTS holdings;")
                cur.execute("""
                    CREATE TABLE holdings (
                        ticker VARCHAR(10),
                        company_name VARCHAR(100), -- The missing column
                        quantity INT,
                        sector VARCHAR(50),
                        region VARCHAR(50),
                        value DECIMAL(10, 2)
                    );
                """)
                cur.execute("""
                    INSERT INTO holdings VALUES 
                    ('AAPL', 'Apple Inc.', 100, 'Technology', 'USA', 17500.00),
                    ('SHEL', 'Shell PLC', 500, 'Energy', 'Europe', 32000.00);
                """)
                conn.commit()
                conn.close()
                st.success("✅ Database Repaired! Please Reload.")
            except Exception as e:
                st.error(f"DB Error: {e}")