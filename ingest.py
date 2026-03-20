from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_chroma import Chroma



load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent / "vector_db")
data_path = Path(__file__).parent/"data"

# PDF-to-text extractor
def pdf_extractor():
    data_path = Path(__file__).parent/"data"

    all_documents = []

    for pdf_file in data_path.glob("*.pdf"):
        elements = partition_pdf(filename=str(pdf_file), strategy="auto", languages=["eng"])
        
        for el in elements:
            if el.text:  
                all_documents.append({
                    "source": pdf_file.name,
                    "text": el.text,
                    "type": el.category  
                })
    return all_documents

        # print(f"{pdf_file.name}: {len(elements)} elements")

#===============================================================================


def create_chunks(all_documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )
    chunks = []
    for doc in all_documents:
        for chunk in text_splitter.split_text(doc["text"]):
            chunks.append({
                "chunk_text": chunk,
                "source": doc["source"],
                "type": doc["type"]
            })

    print(f"Created {len(chunks)} chunks for embeddings")
    return chunks



embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    texts = [chunk["chunk_text"] for chunk in chunks]
    metadatas = [
        {"source": chunk["source"], "type": chunk["type"]}
        for chunk in chunks
    ]

    vectorstore = Chroma.from_texts(
        texts=texts, embedding=embeddings, metadatas=metadatas, persist_directory=DB_NAME
    )
    
    collection = vectorstore._chroma_collection
    count = collection.count()
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")


if __name__ == "__main__":
    all_documents = pdf_extractor()
    chunks = create_chunks(all_documents)
    create_embeddings(chunks)
    print("Ingestion complete")