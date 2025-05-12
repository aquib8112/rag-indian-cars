import modal

image = (
    modal.Image.debian_slim()
    .pip_install(
        "modal>=0.74.0",
        "fastapi",
        "uvicorn",
        "cohere",
        "chromadb",
        "rank_bm25",
        "langchain",
        "jsonlines",
        "langchain-chroma",
        "langchain-cohere",
        "langchain-community",
        "langchain-huggingface",
        "sentence-transformers"
    )
    .add_local_dir(".", "/root") 
)

app  = modal.App("carchatbot", image=image)

@app.function()
@modal.fastapi_endpoint(method="POST",docs=True)
def rag_call(payload: dict):
    print("[Payload]", payload)
    query = payload.get("query", "")
    chat_history = payload.get("chat_history", [])
    from rag_pipeline import response_generation 
    return { "answer": response_generation(query, chat_history)}