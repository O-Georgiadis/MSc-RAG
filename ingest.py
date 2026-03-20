from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# PDF-to-text extractor
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

    # print(f"{pdf_file.name}: {len(elements)} elements")

load_dotenv(override=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


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

chunks = create_chunks(all_documents)
