import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import os

# --- Configuration ---
CLEANED_DATA_PATH = 'data/filtered_complaints.csv'
VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400  # characters
CHUNK_OVERLAP = 50 # characters

# Ensure the vector_store directory exists
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

print("Starting: Chunking, Embedding, and Vector Store Indexing...")

# --- 1. Load Cleaned Data ---
try:
    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Loaded cleaned data from {CLEANED_DATA_PATH}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: {CLEANED_DATA_PATH} not found.")
    exit()

# Prepare data for LangChain Document format
documents = []
for index, row in df.iterrows():
    narrative = str(row['Consumer complaint narrative'])
    if narrative.strip(): # Only add if narrative is not empty after stripping
        documents.append(
            Document(
                page_content=narrative,
                metadata={
                    "complaint_id": row['Complaint ID'],
                    "product": row['Product']
                }
            )
        )

print(f"Created {len(documents)} LangChain Document objects.")


# --- 2. Text Chunking ---
print(f"Applying RecursiveCharacterTextSplitter with chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,  # Use character length
    separators=["\n\n", "\n", " ", ""] # Prioritize splitting by paragraphs, then lines, then words, then characters
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)
print(f"Generated {len(chunks)} chunks from the narratives.")

# --- 3. Choose an Embedding Model ---
print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# --- 4. Embedding and Indexing ---
print("Creating FAISS vector store and generating embeddings...")
# Create a FAISS index from the chunks and embeddings
db = FAISS.from_documents(chunks, embeddings)
print("FAISS vector store created.")

# Save the vector store
db.save_local(VECTOR_STORE_PATH)
print(f"Vector store persisted to {VECTOR_STORE_PATH}/")

