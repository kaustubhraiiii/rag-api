Personal Profile RAG API
A local RAG (Retrieval-Augmented Generation) application that stores personal profiles in a vector database and answers questions about them using a local LLM.
Built with FastAPI, ChromaDB, and Ollama.
How It Works

Store — Profile text is split into chunks and embedded into ChromaDB using nomic-embed-text
Retrieve — Queries are semantically matched against stored chunks
Generate — Relevant chunks are passed as context to a local LLM (qwen2.5:0.5b) to produce answers

Tech Stack

FastAPI — API framework
ChromaDB — Vector database for embedding storage and similarity search
Ollama — Local LLM and embedding model runtime
Pydantic — Request validation

Setup
Prerequisites

Python 3.10+
Ollama installed and running

Install
bashpip install fastapi uvicorn chromadb ollama pydantic
Pull the required Ollama models:
bashollama pull nomic-embed-text
ollama pull qwen2.5:0.5b
Run
Seed the knowledge base (optional):
bashpython build_knowledge_base.py
Start the API:
bashuvicorn main:app --reload
Open http://localhost:8000/docs for the interactive Swagger UI.
Endpoints
MethodEndpointDescriptionPOST/documentsAdd a new user profilePUT/documentsReplace an existing user profileGET/usersList all stored users and chunk countsGET/searchSemantic search without LLM generationGET/askAsk a question with RAG-powered LLM answer
Example Usage
Add a profile:
bashcurl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"user_name": "jordan", "content": "My name is Jordan. I work as a data analyst.\n\nI specialize in Python, SQL, and Pandas."}'
Ask a question:
bashcurl "http://localhost:8000/ask?question=what+does+jordan+do&user=jordan"
Semantic search:
bashcurl "http://localhost:8000/search?query=python+skills&n_results=3"
