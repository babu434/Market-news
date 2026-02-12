import streamlit as st
import os
import pandas as pd
import plotly.express as px
import yfinance as yf
import auth  # Import the security file you just created
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import create_engine, text

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Secure Portfolio", layout="wide")

# Initialize Session State for Login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_full_name" not in st.session_state:
    st.session_state.user_full_name = ""

# Load Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
if "NEON_DB_URL" in st.secrets:
    os.environ["NEON_DB_URL"] = st.secrets["NEON_DB_URL"]

# --- 2. LOGIN PAGE LOGIC ---
def login_page():
    st.title("🔒 Portfolio Login")
    
    # Ensure table exists
    auth.create_users_table()
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            user = st.text_input("Username")
            pw = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In")
            
            if submitted:
                full_name = auth.authenticate_user(user, pw)
                if full_name:
                    st.session_state.authenticated = True
                    st.session_state.user_full_name = full_name
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with tab2:
        with st.form("reg_form"):
            new_user = st.text_input("New Username")
            new_pw = st.text_input("New Password", type="password")
            new_name = st.text_input("Full Name")
            reg_submit = st.form_submit_button("Register")
            
            if reg_submit:
                if len(new_pw) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success, msg = auth.register_user(new_user, new_pw, new_name)
                    if success:
                        st.success("Account created! Please log in.")
                    else:
                        st.error(msg)

# --- 3. MAIN APP LOGIC (The Portfolio) ---
def main_app():
    # Sidebar Logout
    with st.sidebar:
        st.write(f"👤 **{st.session_state.user_full_name}**")
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.rerun()
        st.divider()

    # --- AI & DB SETUP ---
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    try:
        db_url = os.environ["NEON_DB_URL"].replace("postgres://", "postgresql://")
        engine = create_engine(db_url)
        db = SQLDatabase(engine)
        sql_agent = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)
        db_status = "✅ Connected"
    except Exception as e:
        db_status = f"❌ Error: {e}"
        sql_agent = None
        engine = None

    try:
        vector_store = PineconeVectorStore(index_name="market-news", embedding=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    except:
        retriever = None

    # --- HELPER FUNCTIONS ---
    def get_portfolio_df():
        if not engine: return pd.DataFrame()
        df = pd.read_sql("SELECT ticker, quantity, sector FROM holdings", engine)
        if df.empty: return df
        
        current_prices = []
        total_values = []
        for index, row in df.iterrows():
            try:
                price = yf.Ticker(row['ticker']).fast_info['last_price']
            except:
                price = 0.0
            current_prices.append(price)
            total_values.append(price * row['quantity'])
        
        df['current_price'] = current_prices
        df['total_value'] = total_values
        return df

    def get_financial_advice(query, portfolio_df):
        if not portfolio_df.empty:
            portfolio_context = portfolio_df.to_string(index=False)
        else:
            portfolio_context = "Portfolio is empty."
            
        try:
            docs = retriever.invoke(query)
            news_context = "\n".join([d.page_content for d in docs])
        except:
            news_context = "No news found."

        final_prompt = f"""
        You are a financial advisor.
        REAL-TIME PORTFOLIO: {portfolio_context}
        MARKET NEWS: {news_context}
        USER QUESTION: {query}
        Instructions: Use the 'total_value' column for worth. Be concise.
        """
        return llm.invoke(final_prompt).content

    # --- UI LAYOUT ---
    st.title("💰 Talk to Your Portfolio (Pro)")

    df_portfolio = get_portfolio_df()

    # Metrics
    col1, col2, col3 = st.columns(3)
    if not df_portfolio.empty:
        total_net_worth = df_portfolio['total_value'].sum()
        top_stock = df_portfolio.loc[df_portfolio['total_value'].idxmax()]
        col1.metric("Total Net Worth", f"${total_net_worth:,.2f}")
        col2.metric("Top Holding", f"{top_stock['ticker']}", f"${top_stock['total_value']:,.2f}")
        col3.metric("Status", db_status)
    else:
        col1.metric("Total Net Worth", "$0.00")

    # Chart & Chat
    c1, c2 = st.columns([2, 1])
    with c1:
        if not df_portfolio.empty:
            fig = px.bar(df_portfolio, x='ticker', y='total_value', color='sector', title="Live Portfolio Value")
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

        if prompt := st.chat_input("Should I sell Apple?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    response = get_financial_advice(prompt, df_portfolio)
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar Tools
    with st.sidebar:
        st.header("⚡ Trade Simulator")
        with st.form("trade_form"):
            ticker_input = st.text_input("Ticker").upper()
            quantity_input = st.number_input("Shares", min_value=1, value=10)
            action = st.selectbox("Action", ["BUY", "SELL"])
            if st.form_submit_button("Execute"):
                if engine:
                    try:
                        exists_query = text("SELECT quantity FROM holdings WHERE ticker = :t")
                        with engine.connect() as conn:
                            result = conn.execute(exists_query, {"t": ticker_input}).fetchone()
                            if action == "BUY":
                                if result:
                                    new_qty = result[0] + quantity_input
                                    conn.execute(text("UPDATE holdings SET quantity = :q WHERE ticker = :t"), {"q": new_qty, "t": ticker_input})
                                else:
                                    conn.execute(text("INSERT INTO holdings (ticker, quantity, sector) VALUES (:t, :q, 'Unknown')"), {"t": ticker_input, "q": quantity_input})
                                conn.commit()
                                st.success("Trade Executed!")
                                st.rerun()
                            elif action == "SELL":
                                if not result or result[0] < quantity_input:
                                    st.error("Insufficient shares.")
                                else:
                                    new_qty = result[0] - quantity_input
                                    if new_qty == 0:
                                        conn.execute(text("DELETE FROM holdings WHERE ticker = :t"), {"t": ticker_input})
                                    else:
                                        conn.execute(text("UPDATE holdings SET quantity = :q WHERE ticker = :t"), {"q": new_qty, "t": ticker_input})
                                    conn.commit()
                                    st.success("Sold!")
                                    st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

# --- 4. EXECUTION CONTROL ---
if st.session_state.authenticated:
    main_app()
else:
    login_page()