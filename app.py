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
    # Step A: Intent Classification
    router_prompt = f"""
    Classify the query: '{query}' into one of these categories:
    - SQL: Questions about my current portfolio, quantity, value, or holdings.
    - VECTOR: Questions about news, market trends, or opinions.
    - HYBRID: Questions asking how news affects my portfolio.
    Return ONLY the word: SQL, VECTOR, or HYBRID.
    """
    
    # --- ERROR HANDLING BLOCK ---
    try:
        intent = llm.invoke(router_prompt).content.strip()
    except Exception as e:
        st.error(f"⚠️ Groq Error: {str(e)}")
        return "System Error"
    # -----------------------------
    
    response = ""
    
    # Step B: Execution
    try:
        if intent == "SQL":
            response = sql_agent.invoke(query)['output']
            
        elif intent == "VECTOR":
            docs = retriever.invoke(query)
            context = "\n".join([d.page_content for d in docs])
            response = llm.invoke(f"Answer based on news: {context}\nQuestion: {query}").content
            
        elif intent == "HYBRID":
            sql_query = "Summarize my holdings by sector and region."
            portfolio_context = sql_agent.invoke(sql_query)['output']
            
            docs = retriever.invoke(query)
            news_context = "\n".join([d.page_content for d in docs])
            
            final_prompt = f"""
            You are a financial advisor.
            My Portfolio: {portfolio_context}
            Market News: {news_context}
            User Question: {query}
            Analyze the impact of the news on my specific holdings.
            """
            response = llm.invoke(final_prompt).content
            
    except Exception as e:
        st.error(f"⚠️ Agent Error: {str(e)}")
        response = "I encountered an error processing your request."
        
    return response

# 6. STREAMLIT UI
st.title("💰 Talk to Your Portfolio (Free Edition)")
user_query = st.text_input("Ask about your holdings or market news:")

if user_query:
    with st.spinner("Analyzing..."):
        answer = get_financial_advice(user_query)
        st.write(answer)

with st.sidebar:
    st.header("⚠️ Admin Tools")
    if st.button("Run Seed Script (Upload News)"):
        with st.spinner("Seeding Pinecone with fake news..."):
            try:
                # 1. Initialize Pinecone
                import time
                from pinecone import Pinecone, ServerlessSpec
                pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
                
                INDEX_NAME = "market-news"
                
                # Create Index if it doesn't exist
                if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
                    st.write(f"Creating index: {INDEX_NAME}...")
                    pc.create_index(
                        name=INDEX_NAME,
                        dimension=768,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                    time.sleep(10) # Wait for initialization
                
                # 2. Define Data
                data = [
                    {"text": "Apple (AAPL) just released the Vision Pro headset. Analysts predict it will add $5B to revenue in 2025.", "source": "TechCrunch"},
                    {"text": "Shell (SHEL) reports record profits due to rising oil prices. The energy sector is outperforming the S&P 500.", "source": "Bloomberg"},
                    {"text": "The Federal Reserve is cutting interest rates, which is bullish for Tech stocks like Apple.", "source": "WSJ"}
                ]
                
                # 3. Upload
                st.write("Uploading embeddings...")
                from langchain_pinecone import PineconeVectorStore
                PineconeVectorStore.from_texts(
                    texts=[d["text"] for d in data],
                    embedding=embeddings,
                    index_name=INDEX_NAME
                )
                st.success("✅ Success! News uploaded to Pinecone.")
            except Exception as e:
                st.error(f"Error seeding data: {e}")