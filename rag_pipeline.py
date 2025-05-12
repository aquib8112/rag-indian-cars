import os
import re
import json
import cohere
import jsonlines
from copy import deepcopy
from langsmith import Client
from dotenv import load_dotenv
from collections import defaultdict
from langchain_chroma import Chroma
from pydantic import BaseModel
from langchain_cohere import ChatCohere
from typing import Dict, List, Optional, Any
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv() 

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

cohere_api = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(cohere_api)
client = Client()

embedding_retriever = None

_vectorstore = None
_embedding_retriever = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        import os, zipfile

        zip_path = "/root/chroma_db.zip"
        extract_path = "/root/chroma_db"
        if not os.path.exists(extract_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        _vectorstore = Chroma(persist_directory=extract_path, embedding_function=embedding_function)
    return _vectorstore

def get_embedding_retriever():
    global _embedding_retriever
    if _embedding_retriever is None:
        _embedding_retriever = get_vectorstore().as_retriever()
    return _embedding_retriever

bm25_docs = []
with jsonlines.open("bm25_search_docs.jsonl") as reader:
    for obj in reader:
        bm25_docs.append(Document(page_content=obj["page_content"], metadata=obj["metadata"]))

with open("car_names.json", "r") as f:
    car_names = json.load(f)

def bm25_search(queries: List[str], car_names: Optional[List[str]] = None) -> List[Document]:
    """ Performs BM25-based retrieval on a filtered set of documents.

    Args: queries (List[str]): The list of queries to retrieve documents for.
          car_names (Optional[List[str]]): List of car model names to filter documents by.
                                         If not provided or empty, the function returns an empty list.

    Returns: List[Document]: Retrieved documents from the filtered set using BM25 ranking.
                        Returns an empty list if car_names is None or empty."""
    
    if not car_names:
        return []

    filtered_docs = [
        doc for doc in bm25_docs
        if doc.metadata.get("car_model") in car_names
    ]

    retriever = BM25Retriever.from_documents(filtered_docs)
    retriever.k = 5

    results = []
    for q in queries:
        results.extend(retriever.invoke(q))

    return results

def embedding_search(queries: List[str], car_names: Optional[List[str]] = None) -> List[Document]:
    """ Performs embedding-based retrieval with optional filtering by car model(s).

    Args: queries (List[str]): List of queries to search for.
          car_names (Optional[List[str]]): A list of car models to filter results by.
                                         If not provided or empty, no filtering is applied.

    Returns: List[Document]: List of retrieved documents using the embedding-based retriever. """
    
    embedding_retriever = get_embedding_retriever()
    results = []
    filter_ = {"car_model": {"$in": car_names}} if car_names else None

    for q in queries:
        docs = embedding_retriever.invoke(q, filter=filter_, k=5)
        results.extend(docs)
    return results

def deduplicator(docs: List[Document]) -> List[Document]:
    """Removes duplicate documents based on 'chunk_id' in metadata.

    Args: docs (List[Document]): List of Document objects to deduplicate.

    Returns: List[Document]: List of unique Document objects."""
    
    seen_ids = set()
    unique_docs = []

    for doc in docs:
        chunk_id = doc.metadata["chunk_id"]
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            unique_docs.append(doc)

    return unique_docs

def rerank_and_cite(query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
    """ Reranks documents based on relevance to a query and adds citation information to each document.

    Args:query (str): The input query to rerank documents for.
         docs (List[Document]): A list of Document objects to be reranked.
         top_k (int, optional): The number of top documents to return. Defaults to 5.

    Returns: List[Document]: List of reranked Document objects with additional metadata for relevance score and citation ID."""
    
    passages = [doc.page_content for doc in docs]

    response = cohere_client.rerank(
        query=query,
        documents=passages,
        top_n=min(top_k, len(passages)),
        model = "rerank-english-v3.0"
    )

    ranked_docs = []
    for i, item in enumerate(response.results, start=1):  # Start from 1 for human-readable citation ID
        idx = item.index
        doc = deepcopy(docs[idx])
        doc.metadata["relevance_score"] = item.relevance_score
        doc.metadata["citation_id"] = i  # e.g. 1, 2, 3...
        ranked_docs.append(doc)

    return ranked_docs

def extract_car_names_from_query(query: str, car_names_list: List[str]) -> List[str]:
    """ Extracts car names from a query string based on a provided list of car names.

    Args: query (str): The user query from which to extract car names.
          car_names_list (List[str]): A list of known car names to match against.

    Returns: List[str]: A list of car names found in the query. """
    
    car_map = {name.lower(): name for name in car_names_list}
    sorted_names = sorted(car_map.keys(), key=len, reverse=True)
    pattern = r'(?i)\b(' + '|'.join(re.escape(name) for name in sorted_names) + r')\b'
    matches = re.findall(pattern, query)
    
    seen = set()
    result = []
    for match in matches:
        key = match.lower()
        if key not in seen and key in car_map:
            seen.add(key)
            result.append(car_map[key])
    
    return result

def multi_queries(query: str, chat_history: List) -> List[str]:
    class multi_query(BaseModel):
        queries: List[str]

    def history_to_text(chat_history):
        history_lines = []
        for msg in chat_history:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_lines.append(f"{role}: {msg.content}")
        return "\n".join(history_lines)


    multi_query_prompt_template = PromptTemplate.from_template("""
You are a helpful assistant that rewrites and expands user queries using the chat history for context.

Previous chat history:
{chat_history}

Current user query:
"{question}"

Instructions:
1. If the original query already mentions car names, do not change it at all. Keep the query as is and generate 4 alternative versions using slightly different wording.
2. If the query **DOES NOT MENTION A CAR NAME, INFER THE LIKELY CAR NAME FROM THE CHAT HISTORY and REWRITE** the original query to include it, while preserving its intent. Then generate 4 alternative versions based on this revised query.
3. All rewrites should reflect the same intent as the original, with slight wording changes or clarifications where helpful.
4. Return a Python list of exactly 5 strings inside the `queries` field.
""")

    llm = ChatCohere(
        model="command-r-plus",
        temperature=0.1,
        cohere_api_key= cohere_api
    )
    structured_llm = llm.with_structured_output(multi_query)
    chain = multi_query_prompt_template | structured_llm

    context_text = history_to_text(chat_history)

    result = chain.invoke({
        "question": query,
        "chat_history": context_text
    })
    
    return result.queries

def convert_chat_history(chat_history):
    messages = []
    if not isinstance(chat_history, list):
        return messages  # Return empty if it's not a list

    for msg in chat_history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
    return messages

chat_prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a car enthusiast. Search the context carefully for information that answers the question. Give human like answer. Only say 'I do not have information regarding this' if the context lacks the answer."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "Question: {question}")
])

model = ChatCohere(
    model="command-r-plus",
    temperature=0.1,
    cohere_api_key = cohere_api,
)

def cohere_with_docs(inputs):
    messages = chat_prompt_template.format_messages(
        question=inputs["question"],
        chat_history=inputs["chat_history"]
    )
    # Pass messages and documents separately
    response = model.invoke(messages, documents=inputs["documents"])
    return response

def relevant_doc_retriever(inputs: dict) -> dict:

    """ Retrieve and rerank relevant documents based on a user query and chat history.

This function performs multi-query expansion, retrieves documents from BM25 and embedding-based retrievers, 
removes duplicates, reranks the results using Cohere, and formats them for further use.

Parameters:
-----------
inputs : dict
    A dictionary with the following keys:
    - "query" (str): The user's current query.
    - "chat_history" (list, optional): List of previous chat messages (HumanMessage, AIMessage, etc.).
      If not provided, defaults to an empty list.

Returns:
--------
dict:
    A dictionary containing:
    - "documents": A list of top reranked document dicts, each with:
        - "text": The document content.
        - "id": The citation ID (as a string).
        - "chunk_id": The unique chunk identifier (as a string).
        - "relevance_score": Relevance score assigned by the reranker (as a string).
    - "question": The main query used for retrieval and reranking.
    - "chat_history": The (possibly truncated) chat history passed in or inferred. """

    query = inputs["query"]
    chat_history = inputs.get("chat_history", [])

    queries = multi_queries(query, chat_history)
    car_name = extract_car_names_from_query(queries[0], car_names)
    bm25_results = bm25_search(queries, car_name)
    embedding_results = embedding_search(queries, car_name)
    all_results = bm25_results + embedding_results
    deduped_docs = deduplicator(all_results)
    reranked_docs = rerank_and_cite(queries[0], deduped_docs)
    
    cohere_docs = [
    {
        "text": doc.page_content,
        "id": str(doc.metadata["citation_id"]),
        "chunk_id": str(doc.metadata["chunk_id"]),
        "relevance_score": str(doc.metadata["relevance_score"]),
    } for doc in reranked_docs ]
    
    return {
        "documents": cohere_docs,
        "question": queries[0],
        "chat_history": chat_history
    }

chain = RunnableLambda(relevant_doc_retriever) | RunnableLambda(cohere_with_docs)

def format_citations_by_doc(result: AIMessage) -> dict:
    if not hasattr(result, "response_metadata") or "citations" not in result.response_metadata:
        print("No citations found in the response.")
        return {}

    citations = result.response_metadata["citations"]
    doc_map = defaultdict(lambda: {
        "text": "",
        "citations": set(),
        "chunk_id": None,
        "relevance_score": None
    })

    for citation in citations:
        for source in citation.sources:
            doc = source.document
            doc_id = doc.get("id") or doc.get("doc_id") or "unknown_doc"
            doc_text = doc.get("text", "")
            chunk_id = doc.get("chunk_id")
            relevance_score = doc.get("relevance_score")

            # Initialize or update doc_map entry
            doc_map[doc_id]["text"] = doc_text  # Overwrites but should be same for same doc_id
            doc_map[doc_id]["citations"].add(citation.text)
            doc_map[doc_id]["chunk_id"] = chunk_id
            doc_map[doc_id]["relevance_score"] = relevance_score

    # Convert sets to lists for JSON-compatibility
    formatted = {
        doc_id: {
            "text": data["text"],
            "citations": list(data["citations"]),
            "chunk_id": data["chunk_id"],
            "relevance_score": data["relevance_score"]
        }
        for doc_id, data in doc_map.items()
    }

    return formatted

def logged(query: str, response: str, citations: Dict[str, Dict[str, Any]]) -> None:
    ids = []
    relevance_rank = 0
    
    for k, v in citations.items():
        try:
            ids.append(int(k))
        except ValueError:
            ids.append(0)  # fallback for dummy id

        relevance_rank += float(v.get('relevance_score', 0))

    precision = len(ids) / 5 if ids else 0
    relevance_rank = relevance_rank / len(ids) if ids else 0

    shortened_docs = {}
    for doc_id, doc_info in citations.items():
        shortened_docs[doc_id] = {
            "text": doc_info["text"][:200] + "...",
            "citations": doc_info["citations"],
            "chunk_id": doc_info["chunk_id"],
        }

    inputs = [{"query": query}]
    outputs = [{
        "response": response,
        "citations": shortened_docs,
        "Precisionat5": precision,
        "relevance_rank": relevance_rank
    }]

    client.create_examples(
        dataset_id= os.getenv("LOG_DATASET_ID"),
        inputs=inputs,
        outputs=outputs
    )

def response_generation(query: str, chat_history: list = None):
    result = chain.invoke({
        "query": query,
        "chat_history": convert_chat_history(chat_history)
    })
    
    print(result.content,end = '\n\n')
    
    logged(query,result.content,format_citations_by_doc(result))
    return result.content
