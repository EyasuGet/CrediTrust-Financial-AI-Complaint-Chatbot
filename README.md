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

# EDA Summary Report
* The Exploratory Data Analysis (EDA) on the CFPB complaint dataset revealed critical insights into the structure and quality of the raw customer feedback. Initially, the dataset contained a large number of missing values across several columns, most notably in the 'Consumer complaint narrative' (over 6.6 million missing entries), 'Company public response', and 'Tags'. This underscores the necessity of a robust preprocessing phase to ensure data quality for the AI tool.

* An analysis of product distribution showed a wide array of financial products. However, for CrediTrust Financial, the focus was narrowed to 'Credit card', 'Personal loan', 'Buy Now, Pay Later', 'Savings account', and 'Money transfers'. It's important to note that the raw CFPB data doesn't explicitly contain a "Buy Now, Pay Later" product category. To address this, an assumption was made that some existing categories might encompass BNPL, or that this category would be populated from CrediTrust's internal data in a real-world scenario. The distribution across the targeted products in your filtered dataset shows 'Money transfers' leading with over 97,000 complaints, followed by 'Credit card' and 'Personal loan'.

* Further examination of the 'Consumer complaint narrative' revealed a significant number of complaints initially lacked narratives (over 6.6 million), which were subsequently removed. For the remaining narratives, the word count analysis showed a wide distribution, with a mean length of approximately 184 words. While the majority of narratives fall within a manageable length, the presence of very long narratives (up to 6236 words) indicates a need for effective text processing to handle varying levels of detail. The cleaning process, including lowercasing, removing boilerplate text, special characters, and redactions like 'XXXX', was successful in standardizing the narratives, as demonstrated by the sample outputs. This thorough cleaning is crucial for generating high-quality embeddings, which are foundational for the RAG model's ability to accurately retrieve and synthesize relevant information from complaints. The final filtered and cleaned dataset, comprising 263,187 records, is now well-prepared for the subsequent stages of building CrediTrust's complaint-answering chatbot.


### Task 2: Text Chunking, Embedding, and Vector Store Indexing

* **Run the script:**
    ```bash
    python src/chunk_embed_indexing.py
    ```
    *(This script will read `data/filtered_complaints.csv` and save the vector store to `vector_store/`.)*
