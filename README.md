Retrieval-Augmented-Generation (RAG) Chatbot

A minimal, end-to-end RAG pipeline using LangChain that:

pulls a YouTube transcript,

splits it into chunks,

builds a FAISS vector store with OpenAI embeddings, and

answers questions by retrieving relevant chunks and generating a response with an LLM.

Main notebook: Rag_using_langchain.ipynb

✨ Features

Document ingestion from YouTube transcripts

Robust text splitting (overlap to preserve context)

FAISS vector index for fast similarity search

OpenAI embeddings + chat model via LangChain

Clean RAG prompt that refuses when context is insufficient

📁 Project Structure
Retrieval-Augmented-Generation-chatbot/
├── Rag_using_langchain.ipynb     # Complete, runnable notebook
├── README.md                     # This file
└── requirements.txt              # (recommended) see below


Tip: add a .gitignore and do not commit any API keys.

🧰 Requirements

Python 3.9+

A terminal with pip

An OpenAI API key

Create a virtual environment (recommended):

python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1


Install dependencies:

pip install -q youtube-transcript-api langchain-community langchain-openai \
               faiss-cpu tiktoken python-dotenv jupyter


Optional: save them to a file:

pip freeze > requirements.txt

🔐 Configure API Key (safe way)

Create a file named .env in the project root:

OPENAI_API_KEY=your_key_here


Load it inside the notebook:

from dotenv import load_dotenv; load_dotenv()


Security: Never hard-code or commit your API key to GitHub.

🚀 How to Run

Launch Jupyter and open the notebook:

jupyter notebook Rag_using_langchain.ipynb


In the notebook, follow the cells in order. The pipeline has four clear stages:

1) Indexing — Document Ingestion

Fetch the YouTube transcript for a given video ID:

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

video_id = "Gfr50f6ZBvo"  # only the ID, not the full URL

def fetch_transcript(video_id, languages=["en"]):
    try:
        # Fast path (works in most versions)
        return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except AttributeError:
        # Fallback for environments where get_transcript is unavailable
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcripts.find_transcript(languages).fetch()
        return transcript

try:
    transcript_list = fetch_transcript(video_id, languages=["en"])
    transcript_text = " ".join(chunk["text"] for chunk in transcript_list)
except TranscriptsDisabled:
    transcript_text = ""
    print("No captions available for this video.")

2) Indexing — Text Splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.create_documents([transcript_text])

3) Indexing — Embeddings + Vector Store
import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

4) Retrieval + Generation (RAG)
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

prompt = PromptTemplate(
    template=(
        "You are a helpful assistant.\n"
        "Answer ONLY from the provided transcript context.\n"
        "If the context is insufficient, say you don't know.\n\n"
        "{context}\n"
        "Question: {question}"
    ),
    input_variables=["context", "question"],
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def answer_question(q: str):
    retrieved = retriever.invoke(q)
    context = "\n\n".join(d.page_content for d in retrieved)
    final = prompt.format(context=context, question=q)
    return llm.invoke(final).content

# Example
print(answer_question("Is nuclear fusion discussed? What was said?"))

🧪 Example Queries

“Who is Demis Hassabis?”

“Is nuclear fusion discussed? Summarize the points.”

“How is AI used in the conversation?”

🩹 Troubleshooting

AttributeError: YouTubeTranscriptApi has no attribute get_transcript
Use the fallback shown above (list_transcripts(...).find_transcript(...).fetch()).

No transcript found
The video may have captions disabled or only auto-generated in other languages. Try another languages list, e.g. ["en", "en-IN"].

OpenAI authentication error
Ensure .env contains a valid OPENAI_API_KEY and the notebook loaded it via load_dotenv().

🗺️ Roadmap

Add a simple Streamlit/Gradio UI

Support multiple vector DBs (Chroma, Pinecone, etc.)

FastAPI endpoint for production use

📚 References

LangChain Docs – https://python.langchain.com/

youtube-transcript-api – https://github.com/jdepoix/youtube-transcript-api

FAISS – https://faiss.ai/

⚠️ Important Security Note

You accidentally posted an API key in code earlier. Immediately revoke/rotate that key from your OpenAI dashboard and replace it via .env. Never commit secrets to GitHub.

🧩 Quick “How to add this README”

Open your repo → Add file → Create new file

Name it README.md

Paste the content above → Commit

Bas! README tayyar. Agar chaahe to main requirements.txt ka crisp version bhi likh doon:

youtube-transcript-api
langchain-community
langchain-openai
faiss-cpu
tiktoken
python-dotenv
jupyter


Kuch aur polish/add karna ho (badges, screenshots, Streamlit UI), bol de bhai—mein laga deta hoon.






