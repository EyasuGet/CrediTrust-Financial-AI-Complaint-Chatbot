# CrediTrust Financial AI Complaint Chatbot

## Project Overview

This project aims to develop an internal AI tool for CrediTrust Financial to analyze unstructured customer complaint data. Leveraging Retrieval-Augmented Generation (RAG), the tool will enable product managers and non-technical teams to quickly identify major complaint trends and get evidence-backed answers.

## Business Objectives / KPIs

* Decrease the time to identify major complaint trends from days to minutes.
* Empower non-technical teams to get answers without needing a data analyst.
* Shift the company from reacting to problems to proactively identifying and fixing them.

## Project Structure


credittrust_complaint_chatbot/
├── data/
├── notebooks/
├── src/
├── vector_store/
├── .gitignore
├── README.md
└── requirements.txt


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd credittrust_complaint_chatbot
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
## How to Run

### Task 1: Exploratory Data Analysis and Data Preprocessing

* **Using Jupyter Notebook (for interactive exploration):**
    ```bash
    jupyter notebook notebooks/eda_preprocessing.ipynb
    ```
* **Running as a script:**
    ```bash
    python src/eda_preprocessing.py
    ```
    *(Ensure your raw `complaints.csv` is in the `data/` directory or update the script path.)*

### Task 2: Text Chunking, Embedding, and Vector Store Indexing

* **Run the script:**
    ```bash
    python src/chunk_embed_indexing.py
    ```
    *(This script will read `data/filtered_complaints.csv` and save the vector store to `vector_store/`.)*