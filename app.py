import gradio as gr
import pandas as pd
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import os
import google.generativeai as genai
import textwrap
from dotenv import load_dotenv

load_dotenv()

VECTOR_STORE_PATH = '/home/eyuleo/Documents/kifya/week6/vector_store'
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

qa_chain = None
llm_model_name = "Not Loaded" 


def initialize_rag_components():
    """Initializes the RAG components: Vector Store, Retriever, and LLM."""
    global qa_chain, llm_model_name

    if qa_chain is not None:
        print("RAG components already initialized.")
        return "RAG components already initialized."

    print("Initializing RAG components...")

    try:
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            return "Error: GOOGLE_API_KEY environment variable not set. Please set it in your .env file or system environment."
        
        genai.configure(api_key=google_api_key)
        print("Google Generative AI configured for model discovery.")
    except Exception as e:
        return f"Error configuring Google Generative AI. Ensure 'GOOGLE_API_KEY' is set. Error: {e}"

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
        llm_model_name = LLM_MODEL_TO_USE 
        print(f"\nSelected LLM model for RAG: {LLM_MODEL_TO_USE} (Prioritizing Flash for quota reasons)")
    else:
        return "Error: No suitable Gemini model with 'generateContent' support found. Check Google AI Studio and API key permissions."

    print("Loading vector store...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    try:
        db = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Vector store loaded successfully.")
    except Exception as e:
        return f"Error loading vector store: {e}. Ensure Task 2 was completed and 'vector_store' exists in the same directory as app.py."

    retriever = db.as_retriever(search_kwargs={"k": 5})
    print(f"Retriever initialized with k={retriever.search_kwargs['k']} relevant documents.")

    print(f"Initializing Google LLM with model: {LLM_MODEL_TO_USE}...")
    try:
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL_TO_USE, temperature=0.3, google_api_key=google_api_key)
        print("LLM initialized successfully.")
    except Exception as e:
        return f"Error initializing LLM. Check model name and API key. Error: {e}"

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
    return "RAG components initialized successfully!"


def ask_chatbot(query: str, history: list):
    """
    Queries the RAG application and returns the answer along with source documents.
    """
    global qa_chain

    if qa_chain is None:
        return "RAG components not initialized. Please click 'Initialize RAG' first.", []

    try:
        result = qa_chain.invoke({"query": query})

        answer = result["result"]
        source_documents = result["source_documents"]

        sources_display = ""
        if source_documents:
            sources_display += "**Supporting Complaints (Sources):**\n\n"
            for i, doc in enumerate(source_documents):
                wrapped_content = textwrap.fill(doc.page_content, width=100)
                sources_display += (
                    f"**Source {i+1}:**\n"
                    f"- **Complaint ID:** {doc.metadata.get('complaint_id', 'N/A')}\n"
                    f"- **Product:** {doc.metadata.get('product', 'N/A')}\n"
                    f"- **Content Snippet:** \"{wrapped_content[:250]}...\"\n\n"
                )
        else:
            sources_display = "No specific source documents were retrieved for this query."

        history.append((query, answer))
        history.append((None, sources_display))

        return history, sources_display
    except Exception as e:
        error_message = f"An error occurred during the query: {e}"
        history.append((query, error_message))
        return history, f"Error: {e}"

def clear_chat():
    """Clears the chat history."""
    return [], ""

with gr.Blocks(title="CrediTrust Complaint Chatbot (RAG)") as demo:
    gr.Markdown(
        """
        # CrediTrust Financial Complaint Chatbot
        Welcome to the AI tool for analyzing customer complaints!
        Ask me a question about customer complaints related to Credit Cards, Personal Loans, BNPL, Savings Accounts, or Money Transfers.
        I will provide a synthesized answer based on real complaint data.
        """
    )

    status_message = gr.Textbox(label="Status", value="Click 'Initialize RAG' to start.", interactive=False)
    llm_info = gr.Textbox(label="LLM Model In Use", value=llm_model_name, interactive=False)

    initialize_button = gr.Button("Initialize RAG Components")
    initialize_button.click(initialize_rag_components, outputs=status_message)

    chatbot = gr.Chatbot(label="Conversation History", height=400)
    msg = gr.Textbox(label="Your Question", placeholder="e.g., What are common issues with credit card fraud?")
    sources_output = gr.Markdown(label="Retrieved Sources", value="")

    submit_button = gr.Button("Ask CrediTrust")
    clear_button = gr.Button("Clear Chat")

    submit_button.click(
        ask_chatbot,
        inputs=[msg, chatbot],
        outputs=[chatbot, sources_output]
    ).then(lambda: "", inputs=None, outputs=msg) 

    clear_button.click(clear_chat, inputs=None, outputs=[chatbot, sources_output])

    gr.Markdown(
        """
        **Note:** If you encounter `ResourceExhausted` errors, your API quota might be temporarily exceeded.
        Please wait and try again later, or consider upgrading your Google Cloud project.
        """
    )

demo.launch()
