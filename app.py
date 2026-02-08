import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

# 1. SETUP KEYS (In Streamlit, set these in Secrets management)
# os.environ["GROQ_API_KEY"] = "gsk_..."
# os.environ["PINECONE_API_KEY"] = "pc_..."
# os.environ["NEON_DB_URL"] = "postgresql://user:pass@ep-xyz.neondb.net/neondb"
from langchain_pinecone import PineconeVectorStore

# 2. INITIALIZE LLM (GROQ - Free & Fast)
#llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)
#llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0)
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

# 3. SETUP VECTOR STORE (Market News)
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2") # Free local embeddings
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("market-news")
#vector_store = LangchainPinecone(index, embeddings, "text")
#retriever = vector_store.as_retriever(search_kwargs={"k": 3})

vector_store = PineconeVectorStore(
    index_name="market-news", 
    embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 4. SETUP SQL DATABASE (Portfolio)
db = SQLDatabase.from_uri(os.environ["NEON_DB_URL"])
sql_agent = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)

# 5. THE AGENTIC LOGIC (Simplified Router)
def get_financial_advice(query):
    # --- 1. ALWAYS GET PORTFOLIO CONTEXT ---
    # We fetch the portfolio summary first, no matter what the user asks.
    # This ensures the AI never "forgets" what you own.
    try:
        portfolio_context = sql_agent.invoke("Summarize my holdings including company names, quantity, and total value.")['output']
    except Exception as e:
        portfolio_context = "Error retrieving portfolio."

    # --- 2. CHECK FOR NEWS (VECTOR SEARCH) ---
    # We check if the user is asking about news/market trends
    news_context = ""
    try:
        # We retrieve news regardless, just in case the question needs it
        docs = retriever.invoke(query)
        news_context = "\n".join([d.page_content for d in docs])
    except:
        news_context = "No news found."

    # --- 3. THE SYNTHESIS (THE FINAL ANSWER) ---
    # We give the LLM *both* pieces of information and let it decide how to use them.
    final_prompt = f"""
    You are a highly capable financial advisor.
    
    Processing Context:
    1. USER PORTFOLIO: {portfolio_context}
    2. MARKET NEWS: {news_context}
    
    USER QUESTION: {query}
    
    INSTRUCTIONS:
    - If the user asks about their holdings, use the Portfolio data.
    - If the user asks about news, use the Market News data.
    - If the user asks how news affects them (Hybrid), COMBINE the data. 
    - You MUST explicitly mention the user's share counts and values if relevant.
    - Do not say "If you own..." because you have the portfolio data right above.
    """
    
    try:
        response = llm.invoke(final_prompt).content
    except Exception as e:
        response = f"⚠️ Error generating response: {str(e)}"
        
    return response

# 6. STREAMLIT UI
st.title("💰 Talk to Your Portfolio (Free Edition)")
user_query = st.text_input("Ask about your holdings or market news:")

if user_query:
    with st.spinner("Analyzing..."):
        answer = get_financial_advice(user_query)
        st.write(answer)

# --- ADMIN TOOLS ---
with st.sidebar:
    st.divider()
    st.header("⚙️ Admin Tools")
    
    # We add a checkbox to prevent accidental deletion
    force_reset = st.checkbox("Force Re-create Index (Fix 400 Error)")
    
    if st.button("Load Fake News"):
        with st.spinner("Processing..."):
            try:
                import time
                from pinecone import Pinecone, ServerlessSpec
                
                # 1. Connect
                pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
                INDEX_NAME = "market-news"
                
                # 2. DELETE EXISTING INDEX (The Fix for 400 Error)
                existing_indexes = [i.name for i in pc.list_indexes()]
                
                if force_reset and INDEX_NAME in existing_indexes:
                    st.warning(f"Deleting old index '{INDEX_NAME}' to fix dimension mismatch...")
                    pc.delete_index(INDEX_NAME)
                    time.sleep(5) # Wait for deletion
                    existing_indexes = [] # Force creation logic below
                
                # 3. CREATE NEW INDEX (With correct 768 dimensions)
                if INDEX_NAME not in existing_indexes:
                    st.write(f"Creating fresh index '{INDEX_NAME}' (Dimension: 768)...")
                    pc.create_index(
                        name=INDEX_NAME,
                        dimension=768,  # MATCHES HUGGINGFACE MODEL
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    time.sleep(15) # Wait for initialization
                
                # 4. UPLOAD DATA
                news_data = [
                    {"text": "Apple (AAPL) releases the Vision Pro headset. Analysts predict it will add $5B to revenue in 2025.", "source": "TechCrunch"},
                    {"text": "Shell (SHEL) reports record profits due to rising oil prices. The energy sector is outperforming the S&P 500.", "source": "Bloomberg"},
                    {"text": "The Federal Reserve is cutting interest rates, which is bullish for Tech stocks like Apple.", "source": "WSJ"}
                ]
                
                st.write("Embedding data...")
                # Note: We use the 'vector_store' object defined in the main app
                # but we must ensure it's using the *newly created* index.
                # So we re-initialize the store connection here just to be safe:
                from langchain_pinecone import PineconeVectorStore
                
                PineconeVectorStore.from_texts(
                    texts=[d["text"] for d in news_data],
                    embedding=embeddings,
                    index_name=INDEX_NAME,
                    metadatas=[{"source": d["source"]} for d in news_data]
                )
                
                st.success("✅ Success! Index reset and news uploaded.")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")