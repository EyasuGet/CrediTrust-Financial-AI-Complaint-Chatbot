import pandas as pd
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = 'vector_store'
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Starting Task 3: Implementing the RAG Application...")

try:
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        print("Error: GOOGLE_API_KEY environment variable not set. Please set it in your .env file or system environment.")
        exit()
    
    genai.configure(api_key=google_api_key)
    print("Google Generative AI configured for model discovery.")
except Exception as e:
    print(f"Error configuring genai for model discovery. Ensure 'GOOGLE_API_KEY' is set. Error: {e}")
    exit()

LLM_MODEL_TO_USE = None
print("\n--- Listing available Gemini models ---")
available_models = []
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        available_models.append(m.name)

if "models/gemini-1.5-flash" in available_models:
    LLM_MODEL_TO_USE = "models/gemini-1.5-flash"
elif "models/gemini-1.5-flash-latest" in available_models:
    LLM_MODEL_TO_USE = "models/gemini-1.5-flash-latest"
elif "models/gemini-1.5-pro" in available_models:
    LLM_MODEL_TO_USE = "models/gemini-1.5-pro"
elif "models/gemini-1.0-pro" in available_models:
    LLM_MODEL_TO_USE = "models/gemini-1.0-pro"
else:
    for model_name in available_models:
        if "gemini" in model_name:
            LLM_MODEL_TO_USE = model_name
            break

if LLM_MODEL_TO_USE:
    print(f"\nSelected LLM model for RAG: {LLM_MODEL_TO_USE} (Prioritizing Flash for quota reasons)")
else:
    print("\nError: No suitable Gemini model with 'generateContent' support found.")
    print("Please check Google AI Studio for available models and your API key permissions.")
    exit()


# --- 1. Load Vector Store ---
print("Loading vector store...")
embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
try:
    db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    print("Vector store loaded successfully.")
except Exception as e:
    print(f"Error loading vector store: {e}")
    print("Please ensure Task 2 was completed and the 'vector_store' directory exists and is valid.")
    exit()

# --- 2. Initialize Retriever ---
retriever = db.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 relevant documents
print(f"Retriever initialized with k={retriever.search_kwargs['k']} relevant documents.")


# --- 3. Initialize Large Language Model (LLM) ---
print(f"Initializing Google LLM with model: {LLM_MODEL_TO_USE}...")
try:
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_TO_USE, temperature=0.3, google_api_key=google_api_key)
    print("LLM initialized successfully.")
except Exception as e:
    print(f"Error initializing LLM. Check model name and API key. Error: {e}")
    exit()

# --- 4. Set up the RAG Chain with Prompt Engineering ---
template = """You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context: {context}

Question: {question}

Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
print("RAG chain set up.")

# --- 5. Implement a Query Function ---
def ask_credittrust_rag(query: str):
    """
    Queries the RAG application and returns the answer along with source documents.
    """
    print(f"\n--- Processing Query: \"{query}\" ---")
    try:
        result = qa_chain.invoke({"query": query})

        answer = result["result"]
        source_documents = result["source_documents"]

        print(f"\nAI Answer:\n{answer}")

        print("\n--- Retrieved Sources (for evaluation) ---")
        retrieved_sources_info = []
        if source_documents:
            for i, doc in enumerate(source_documents):
                source_info = {
                    "content_snippet": doc.page_content[:150] + "...", # Snippet for brevity
                    "complaint_id": doc.metadata.get('complaint_id', 'N/A'),
                    "product": doc.metadata.get('product', 'N/A')
                }
                retrieved_sources_info.append(source_info)
                print(f"Source {i+1}: [ID: {source_info['complaint_id']}, Product: {source_info['product']}]")
                print(f"  Snippet: \"{source_info['content_snippet']}\"")
                print("-" * 30)
        else:
            print("No source documents were retrieved for this query.")
        
        return answer, retrieved_sources_info
    except Exception as e:
        print(f"An error occurred during the query: {e}")
        return "An error occurred while processing your request.", []

# --- 6. Qualitative Evaluation ---
print("\n--- Starting Qualitative Evaluation ---")

evaluation_questions = [
    "What are the most common complaints about unauthorized transactions on credit cards?",
    "Are there any recurring issues with delayed money transfers?",
    "Summarize the problems customers face with interest rates on personal loans.",
    "What are the main complaints regarding hidden fees in savings accounts?",
    "Can you tell me about issues related to incorrect billing for Buy Now, Pay Later services?",
    "What kind of fraud is being reported across different products?",
    "Are there complaints about poor customer service related to credit cards?",
    "What are the typical reasons for account closures in savings accounts?",
    "Describe issues with loan application rejections for personal loans.",
    "What are the most common complaints about mobile app functionality for money transfers?"
]

evaluation_results = []

for i, q in enumerate(evaluation_questions):
    print(f"\n--- Running Evaluation for Question {i+1}/{len(evaluation_questions)} ---")
    generated_answer, retrieved_sources = ask_credittrust_rag(q)
    
    evaluation_results.append({
        "Question": q,
        "Generated Answer": generated_answer,
        "Retrieved Sources": retrieved_sources,
        "Quality Score": "N/A", 
        "Comments/Analysis": "N/A" 
    })

print("\n--- Qualitative Evaluation Complete ---")

print("\n--- Evaluation Results Table (Copy to Report) ---")
print("| Question | Generated Answer | Retrieved Sources (1-2) | Quality Score (1-5) | Comments/Analysis |")
print("|---|---|---|---|---|")
for res in evaluation_results:
    sources_str = ""
    for j, src in enumerate(res["Retrieved Sources"][:2]):
        sources_str += f"[ID: {src['complaint_id']}, Product: {src['product']}] "
        if j < len(res["Retrieved Sources"][:2]) - 1:
            sources_str += " "
    
    escaped_answer = res["Generated Answer"].replace("|", "\\|")
    
    print(f"| {res['Question']} | {escaped_answer} | {sources_str.strip()} | {res['Quality Score']} | {res['Comments/Analysis']} |")

print("\nTask 3 completed successfully. Please review the output and update the report with the evaluation table and analysis.")

