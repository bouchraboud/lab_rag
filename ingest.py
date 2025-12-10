# ingest.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os, glob, json

def load_and_split(pdf_path, chunk_size=1000, chunk_overlap=200):
    """Charger et découper un PDF en chunks"""
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    print(f"Loaded {len(docs)} pages, splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(docs)
    return split_docs

def main():
    os.makedirs("chunks", exist_ok=True)
    
    pdf_files = glob.glob("data/*.pdf")
    if not pdf_files:
        print("❌ No PDF files found in data/ folder!")
        print("Please download IPCC PDFs first.")
        return
    
    for p in pdf_files:
        print(f"\n{'='*60}")
        print(f"Processing {p}...")
        print('='*60)
        
        try:
            docs = load_and_split(p)
            
            # Sauvegarder les chunks en JSON
            out = [{
                "page_content": d.page_content,
                "metadata": d.metadata
            } for d in docs]
            
            fn = os.path.join("chunks", os.path.basename(p) + ".json")
            with open(fn, "w", encoding="utf8") as f:
                json.dump(out, f, indent=2)
            
            print(f"✅ Saved {len(docs)} chunks to {fn}")
        except Exception as e:
            print(f"❌ Error processing {p}: {e}")
    
    print(f"\n{'='*60}")
    print("✅ Ingestion complete!")
    print('='*60)

if __name__ == "__main__":
    main()