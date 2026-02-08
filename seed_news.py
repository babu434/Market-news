import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIG ---
# Ensure these match your Streamlit secrets or set them here temporarily
# os.environ["PINECONE_API_KEY"] = "pc_..." 

INDEX_NAME = "market-news"

# --- 1. CONNECT & CREATE INDEX ---
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Check if index exists
existing_indexes = [i.name for i in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"Creating index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=768, # Matches the HuggingFace model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(10) # Wait for initialization

# --- 2. THE DATA (News + Apple Context) ---
news_articles = [
    {
        "text": "Apple (AAPL) releases the Vision Pro. Analysts call it a 'spatial computing breakthrough' but worry about the high price point. Tech stocks are rallying.",
        "source": "TechCrunch", 
        "date": "2024-02-15"
    },
    {
        "text": "Shell (SHEL) reports record quarterly profits driven by global energy demand. The company announces a stock buyback program.",
        "source": "Bloomberg", 
        "date": "2024-02-12"
    }
]

# --- 3. UPLOAD ---
print("Embedding and uploading data...")
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
docsearch = PineconeVectorStore.from_texts(
    texts=[t["text"] for t in news_articles],
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("✅ Success! News uploaded.")