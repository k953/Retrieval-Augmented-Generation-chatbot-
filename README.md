Retrieval-Augmented-Generation (RAG) Chatbot

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) based chatbot using LangChain.
RAG allows Large Language Models (LLMs) to answer queries with up-to-date and domain-specific knowledge by retrieving relevant documents from a knowledge base before generating a response.


📌 Features

✅ Uses LangChain for RAG pipeline

✅ Integration with Vector Database (FAISS / Chroma can be used)

✅ Document ingestion and embeddings creation

✅ Context-aware answers by combining retrieval + LLM generation

✅ Example notebook included: Rag_using_langchain.ipynb



⚙️ Installation

Clone this repository and navigate into it:

git clone https://github.com/k953/Retrieval-Augmented-Generation-chatbot.git
cd Retrieval-Augmented-Generation-chatbot


Create a virtual environment (recommended):

python3 -m venv rag_env
source rag_env/bin/activate   # For Linux/Mac
rag_env\Scripts\activate      # For Windows

Install required dependencies:

pip install -r requirements.txt


(You can generate requirements.txt from the notebook by exporting installed packages if needed.)

🚀 Usage

Open the Jupyter Notebook:
jupyter notebook Rag_using_langchain.ipynb
Inside the notebook, follow these steps:

Import Libraries – Load LangChain, vector database, and embedding model.

Load Documents – Add your custom documents (PDFs, text, or knowledge base).

Create Embeddings – Convert text into embeddings using OpenAI or HuggingFace models.

Store in Vector DB – Store embeddings in FAISS/Chroma for similarity search.

Build RAG Chain – Combine retriever with LLM (OpenAI, Llama2, etc.).

Chat – Ask queries and get contextual answers.


📂 Project Structure
Retrieval-Augmented-Generation-chatbot/
│── Rag_using_langchain.ipynb   # Main RAG chatbot notebook
│── requirements.txt            # Dependencies (to be added)
│── README.md                   # Project documentation

🔑 API Keys

This project requires an LLM API key (e.g., OpenAI key).
Set your key in environment variables before running the notebook:

export OPENAI_API_KEY="your_api_key_here"


For Windows (PowerShell):

setx OPENAI_API_KEY "your_api_key_here"

📖 References

LangChain Documentation

RAG Overview by OpenAI

📌 Future Work

Add Streamlit/Gradio interface for live chatbot

Support multiple vector databases (Weaviate, Pinecone, etc.)

Deploy as API using FastAPI






