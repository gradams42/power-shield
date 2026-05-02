import os, shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def index_docs():
    # 1. Start Fresh: Delete old DB to prevent metadata conflicts
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        print("Cleaning old database...")

    # 2. Load and Split
    loader = DirectoryLoader('./manuals', glob="./*.txt", loader_cls=TextLoader)
    docs = loader.load()
    
    # 700 char chunks: small enough to keep R3-FDIA and R4-PM in separate chunks
    # (combined ~790 chars), while still fitting each FDIA protocol in one chunk.
    # Overlap 100 (reduced from 150) to avoid cross-protocol content bleeding.
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 3. Enhanced Metadata Tagging
    # Tag by protocol header (e.g., "PROTOCOL R1-FDIA") not by content scanning.
    # Content scanning incorrectly tags multi-relay protocols like FREQ-01 and
    # EXPECTED-09 as R1 because "R1" appears first in the text.
    import re as _re
    _RELAY_HEADER = _re.compile(r"PROTOCOL\s+(R([1-4])[A-Z0-9\-]+)", _re.IGNORECASE)

    for chunk in chunks:
        match = _RELAY_HEADER.search(chunk.page_content)
        if match:
            chunk.metadata["relay"] = f"R{match.group(2)}"
        else:
            chunk.metadata["relay"] = "GENERIC"

    # 4. Create Vector Store
    embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(
        chunks, 
        embed_model, 
        persist_directory="./chroma_db"
    )
    
    print(f"--- SUCCESS: Indexed {len(chunks)} chunks ---")
    
    # Verification Test
    test = db.similarity_search("Voltage fluctuations", k=1)
    if test:
        print(f"Verified Retrieval. Example Metadata: {test[0].metadata}")

if __name__ == "__main__":
    index_docs()
