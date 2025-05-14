# 🚗 CarSmjho

**CarSmjho** is a Retrieval-Augmented Generation (RAG) system that curates responses using reviews and specification charts of cars available in the Indian market.

## 🎯 Motivation
There were three core reasons behind building this project:
1. **Hallucinations in LLMs** – Reducing the tendency of models to generate inaccurate or made-up content.
2. **Latency of Chat Models** – Addressing the slow response times of models like ChatGPT or Gemini when querying external sources.
3. **Personal Learning Goal** – Exploring and understanding the inner workings of RAG systems.

---

## 🧰 Tech Stack
This project leverages the following libraries and tools:
- **[LangChain](https://www.langchain.com/)** – For constructing RAG pipeline components.
- **[LangSmith](https://www.langchain.com/langsmith)** – For tracing and logging evaluation data.
- **Python** – The core programming language powering the system.
- **[Cohere API](https://docs.cohere.com/)** – Used for multi-query generation and final response generation with citation support.
- **[Hugging Face Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)** – Provides sentence embeddings for vector search.
- **[Modal](https://modal.com/)** – Serverless deployment of the FastAPI endpoint.
- **[Pydantic](https://docs.pydantic.dev/)** – Used for FastAPI request and response validation.
- **[ChromaDB](https://www.trychroma.com/)** – Vector store for storing and retrieving document embeddings.
- **[rank_bm25](https://pypi.org/project/rank-bm25/)** – Implements BM25 for keyword-based sparse retrieval.

---

## 🛠️ Installation
To clone this repository, open your terminal and run:
```bash
git clone https://github.com/aquib8112/rag-indian-cars.git
```
After cloning the repo, navigate into the project directory:
```bash
cd rag-indian-cars
```
To run this project, you will need four things:
1. **`Cohere API Key`** – This is used for multi-query generation and response generation. The reason for choosing Cohere is simple: they have a generous free tier, and their model can directly provide citations.
2. **`LangSmith API Key`** – This helps in visually tracing the system, which can be useful for debugging.
3. **`LangSmith Dataset ID`** – This project also logs the whole response cycle and some metrics, so I highly recommend using a LangSmith dataset to log everything.
4. **`Modal Account`** – Create a Modal account and authenticate your project. I've named the project "carchatbot", but you can change this if you'd like.

Create a `.env` file in the same directory as the downloaded code and add the following:
* LANGCHAIN_API_KEY = your_langsmith_api_key  
* LANGCHAIN_PROJECT = your_project_name_for_langsmith  
* COHERE_API_KEY = your_cohere_api_key  
* LOG_DATASET_ID = your_langsmith_dataset_id

---

## ⚙️ How It Works (Step-by-Step)
1. **User submits a query** via the frontend.  
2. **Multi-Query Generator** (Cohere) creates variations of the question to improve retrieval coverage.  
3. **BM25 Retriever** fetches relevant documents based on keyword overlap.  
4. **ChromaDB** performs vector-based retrieval using embeddings.  
5. **Results from BM25 and ChromaDB** are merged, deduplicated, and filtered using metadata.  
6. **Cohere Re-Ranker** ranks the documents based on relevance to the query.  
7. **Top documents + original query** are passed to the LLM for final response generation.  
8. **LangSmith** logs the query, response, citations, and metrics like relevance and precision for debugging and evaluation.

---

## 📁 File & Folder Descriptions

### `data_creation/`  
Contains all the raw and processed data used to build the RAG pipeline. Includes:

- `data-creation-for-rag.ipynb` – Jupyter notebook that prepares car reviews/specs, generates `bm25_search_docs.jsonl`, and builds the ChromaDB.  
- `cln_car.csv` – Cleaned car dataset with basic details.  
- `car_reviews/` – Plain text files containing customer reviews for different cars.  
- `car_spec/` – Plain text files containing technical specifications of different cars.


### `frntend/`  
The minimal frontend for the chatbot UI. Includes:

- `index.html` – The main HTML structure for the chat interface.  
- `script.js` – JavaScript that handles sending user queries and displaying responses.  
- `style.css` – Basic styling for the chatbot interface.


### `bm25_search_docs.jsonl`  
JSONL file used by the BM25 retriever. Each line is a document chunk created from reviews/specs.


### `car_names.json`  
A helper file mapping various car names (e.g., aliases or lowercase versions) to a standard format used for normalization.


### `chroma_db.zip`  
Zipped version of the persistent Chroma vector database generated during preprocessing. Used directly by the retriever.


### `rag_api.py`  
The entry point for serving the chatbot using Modal. It:
- Defines a container image with required Python dependencies.  
- Exposes an HTTP API (`/rag_call`) using FastAPI via Modal.  
- Accepts user queries and forwards them to the `response_generation` function in `rag_pipeline.py`.


### `rag_pipeline.py`  
Contains the core RAG logic. It:
- Loads the Chroma vectorstore and BM25 retriever.  
- Generates multi-query variations.  
- Fetches and re-ranks relevant document chunks.  
- Adds sources/citations and formats the final answer.  
- Designed to be called by the Modal API wrapper (`rag_api.py`).


### `requirements.txt`  
Lists all Python dependencies needed to run the backend (`cohere`, `langchain`, `chromadb`, etc.).


### `.gitignore`  
Specifies which files/folders Git should ignore — like Python cache and sensitive config.


### `README.md`  
The documentation file you're reading right now. Explains project structure, usage, and setup.
