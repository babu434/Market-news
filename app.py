# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Talk to Your Portfolio", layout="wide")

# Load Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
if "NEON_DB_URL" in st.secrets:
    os.environ["NEON_DB_URL"] = st.secrets["NEON_DB_URL"]

# Initialize AI
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# --- CONNECT TO DATABASE (ROBUST MODE) ---
try:
    # 1. Force the engine to use the 'public' schema
    # This helps when the AI gets lost in system tables
    db = SQLDatabase.from_uri(
        os.environ["NEON_DB_URL"],
        schema="public", 
        sample_rows_in_table_info=3
    )
    
    # 2. DEBUG: Show us exactly what the AI sees
    # This will print a list of tables at the top of your app.
    # If this list is empty, the DB is truly empty.
    table_names = db.get_usable_table_names()
    if not table_names:
        st.warning("⚠️ Database connected, but NO tables found. Please click 'Create/Reset Portfolio DB' in the sidebar.")
    else:
        # Success! Initialize the Agent
        sql_agent = create_sql_agent(llm, db=db, agent_type="tool-calling", verbose=True)
        
except Exception as e:
    st.error(f"❌ Connection Error: {e}")
    db = None
    sql_agent = None

# Connect to News
try:
    vector_store = PineconeVectorStore(index_name="market-news", embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
except:
    retriever = None