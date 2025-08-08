# ingestion_server.py

import os
import fitz  # PyMuPDF
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Use a fast, local embedding model
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
import logging
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv


load_dotenv()

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PINECONE_API_KEY = "pcsk_46Kejg_UpcvvhG3jeGtrGXTucVLXyWMCMGUuTo1xVRanasZ3cE38cLq3zKFm8X2BExAnoB"

PINECONE_INDEX_NAME = "bajaj-ingestion"

VECTOR_DIMENSION = 384 
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set. Please create a .env file or export the variable.")


# --- INITIALIZE SERVICES ---
try:
#  new model use kela fastembeddings : all mini
    embeddings_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize Pinecone Client
    pc = Pinecone(api_key=PINECONE_API_KEY)

except Exception as e:
    logging.error(f"Failed to initialize services: {e}")
    raise

# --- FASTAPI APP ---
app = FastAPI(
    title="Document Ingestion API",
    description="An API to process documents, create embeddings locally, and store them in Pinecone."
)

@app.on_event("startup")
async def startup_event():
    """
    On server startup, check for the Pinecone index and create it if it doesn't exist.
    """
    logging.info("--- Ingestion Server Starting Up ---")
    try:
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            logging.warning(f"Pinecone index '{PINECONE_INDEX_NAME}' not found. Creating it now with dimension {VECTOR_DIMENSION}...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=VECTOR_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            logging.info(f"Successfully created index: '{PINECONE_INDEX_NAME}'")
        else:
           
            index_description = pc.describe_index(PINECONE_INDEX_NAME)
            if index_description.dimension != VECTOR_DIMENSION:
                 logging.error(f"CRITICAL: Existing Pinecone index '{PINECONE_INDEX_NAME}' has dimension {index_description.dimension}, but script requires {VECTOR_DIMENSION}. Please delete the existing index or update the script.")
             
                 raise ValueError("Mismatched Pinecone index dimension.")
            else:
                logging.info(f"Target Pinecone index '{PINECONE_INDEX_NAME}' already exists with correct dimension.")

        logging.info("Ready to receive PDF files at the /ingest endpoint.")

    except Exception as e:
        logging.error(f"Could not connect to or create Pinecone index: {e}")
        raise

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Endpoint to ingest a PDF document. It extracts text, splits it into chunks,
    generates embeddings locally, and upserts them into a Pinecone index.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    try:
        logging.info(f"Received file: {file.filename}. Starting ingestion process...")
        file_content = await file.read()

        # Step 1: Extract text from PDF using PyMuPDF
        logging.info("Step 1: Extracting text from PDF...")
        extracted_pages = []
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text:  # Only process pages with actual text
                    extracted_pages.append(Document(
                        page_content=text,
                        metadata={"source": file.filename, "page": page_num + 1}
                    ))
        
        if not extracted_pages:
            logging.warning(f"File '{file.filename}' has no extractable text content.")
            raise HTTPException(status_code=400, detail="The provided document has no text content.")
        logging.info(f"Extracted text from {len(extracted_pages)} pages.")

        # Step 2: Split document into smaller chunks for effective embedding
        logging.info("Step 2: Splitting document into smaller chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs_chunks = text_splitter.split_documents(extracted_pages)
        logging.info(f"Split into {len(docs_chunks)} chunks.")

        # Step 3: Embed chunks and upload to Pinecone
        logging.info(f"Step 3: Embedding {len(docs_chunks)} chunks locally and uploading to Pinecone...")
        PineconeVectorStore.from_documents(
            documents=docs_chunks,
            embedding=embeddings_model,
            index_name=PINECONE_INDEX_NAME
        )
        logging.info("--- Ingestion Complete ---")
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": f"Document processed and stored in Pinecone index '{PINECONE_INDEX_NAME}'.",
            "chunk_count": len(docs_chunks)
        }

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions to let FastAPI handle the response
        raise http_exc
    except Exception as e:
        logging.error(f"An unexpected error occurred during ingestion for file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {e}")

# --- HOW TO RUN THIS SERVER ---
# 1. Create a file named .env in the same directory and add your key:
#    PINECONE_API_KEY="your_pinecone_key_here"
#
# 2. Make sure you have installed the required packages:
#    pip install "fastapi[all]" uvicorn pymupdf langchain-pinecone pinecone-client python-dotenv langchain-community fastembed
#
# 3. In your terminal, run the following command:
#    uvicorn ingestion_server:app --reload --port 8001
# Custom Build command : uvicorn app:app --host 0.0.0.0 --port 8000 --reload