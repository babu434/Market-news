import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIG ---
# Ensure your keys are set in your environment or paste them here
# os.environ["PINECONE_API_KEY"] = "pc_..."

INDEX_NAME = "market-news"

# 1. Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Create Index if not exists
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    print(f"Creating index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(10) # Wait for it to be ready

# 2. Define Fake News
data = [
    {"text": "Apple (AAPL) just released the Vision Pro headset. Analysts predict it will add $5B to revenue in 2025.", "source": "TechCrunch"},
    {"text": "Shell (SHEL) reports record profits due to rising oil prices. The energy sector is outperforming the S&P 500.", "source": "Bloomberg"},
    {"text": "The Federal Reserve is cutting interest rates, which is bullish for Tech stocks like Apple.", "source": "WSJ"}
]

# 3. Upload to Vector DB
print("Uploading news...")
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
PineconeVectorStore.from_texts(
    texts=[d["text"] for d in data],
    embedding=embeddings,
    index_name=INDEX_NAME
)
print("✅ Success! News uploaded.")