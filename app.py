# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="RAG IPCC Climate API")

# Initialize embeddings
print("Loading vector database...")
embedding_fn = OllamaEmbeddings(model="nomic-embed-text:latest")

# Load vector database
vectordb = Chroma(
    persist_directory="vectordb",
    embedding_function=embedding_fn
)

# Create retriever
retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Initialize LLM
print("Initializing language model...")
# Make sure to use a model you have installed
llm = ChatOllama(model="llama3.2:latest", temperature=0.0)

# Prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions about climate science based on IPCC reports. "
               "Use only the provided context to answer questions. "
               "If the answer is not in the context, say 'I don't know based on the provided documents.'"),
    ("user", """Context:
{context}

Question: {question}""")
])

print("✓ RAG system ready!")

class QueryIn(BaseModel):
    question: str

@app.post("/ask")
async def ask(q: QueryIn):
    print(f"\nQuery: {q.question}")

    # Retrieve documents (pass run_manager=None for compatibility)
    docs = retriever._get_relevant_documents(q.question, run_manager=None)

    # Combine context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Prepare prompt
    messages = prompt_template.format_messages(
        context=context,
        question=q.question
    )

    # Run LLM in background thread to avoid blocking
    response = await run_in_threadpool(lambda: llm.invoke(messages))

    # Collect sources
    sources = [
        {"content": doc.page_content[:200] + "...", "metadata": doc.metadata}
        for doc in docs
    ]

    return {
        "answer": response.content,
        "sources": sources
    }

@app.get("/")
def root():
    return {
        "message": "RAG IPCC Climate API",
        "endpoints": {
            "/ask": "POST - Ask questions about IPCC reports",
            "/docs": "GET - API documentation"
        }
    }
