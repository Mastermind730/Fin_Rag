# retriever.py
import os
from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "bajaj-ingestion"

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Same embedding model as ingestion
embeddings_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to existing Pinecone vector store
vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings_model
)

# Export retriever for other files to import
retriever = vector_store.as_retriever()
