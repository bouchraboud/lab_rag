# embeddings.py
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import json
import os
import time

def embed_and_store(chunks_dir="chunks", persist_directory="vectordb"):
    """Load chunks, create embeddings, and store in vector database."""
    
    print("Initializing embeddings model...")
    embedder = OllamaEmbeddings(
        model="nomic-embed-text:latest",
        base_url="http://localhost:11434"
    )
    
    # Load all chunk files
    print(f"Loading chunks from {chunks_dir}/...")
    documents = []
    
    chunk_files = [f for f in os.listdir(chunks_dir) if f.endswith(".json")]
    
    if not chunk_files:
        print(f"ERROR: No chunk files found in {chunks_dir}/")
        print("Please run ingest.py first!")
        return
    
    for filename in chunk_files:
        filepath = os.path.join(chunks_dir, filename)
        print(f"  Loading {filename}...")
        
        with open(filepath, "r", encoding="utf8") as f:
            items = json.load(f)
        
        for item in items:
            doc = Document(
                page_content=item["page_content"],
                metadata=item.get("metadata", {})
            )
            documents.append(doc)
    
    print(f"\n✓ Loaded {len(documents)} chunks total")
    print(f"Creating embeddings and storing in vector database...")
    print("  (Processing in batches to avoid connection issues...)")
    
    # Process in smaller batches to avoid connection timeouts
    batch_size = 50  # Reduced batch size for Windows
    vectordb = None
    
    total_batches = (len(documents) - 1) // batch_size + 1
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_num = i // batch_size + 1
        
        print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        try:
            if vectordb is None:
                # Create initial database with first batch
                vectordb = Chroma.from_documents(
                    documents=batch,
                    embedding=embedder,
                    persist_directory=persist_directory
                )
            else:
                # Add to existing database
                vectordb.add_documents(batch)
            
            # Small delay between batches to avoid overwhelming Ollama
            if i + batch_size < len(documents):
                time.sleep(1)
                
        except Exception as e:
            print(f"  ERROR on batch {batch_num}: {e}")
            print(f"  Retrying in 5 seconds...")
            time.sleep(5)
            
            # Retry once
            try:
                if vectordb is None:
                    vectordb = Chroma.from_documents(
                        documents=batch,
                        embedding=embedder,
                        persist_directory=persist_directory
                    )
                else:
                    vectordb.add_documents(batch)
            except Exception as e2:
                print(f"  FAILED after retry: {e2}")
                print(f"  Continuing with next batch...")
                continue
    
    if vectordb:
        print(f"\n✓ Vector database created at {persist_directory}/")
        print(f"✓ Indexed {len(documents)} document chunks")
    else:
        print("\n✗ Failed to create vector database")
    
    return vectordb

if __name__ == "__main__":
    embed_and_store()