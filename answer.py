from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma



MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent / "vector_db")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a helpful and precise assistant answering questions based on provided context from PDF documents.

Your task is to answer the user's question using ONLY the information from the retrieved context.

Guidelines:
- Use the provided context as your primary source of truth.
- If the answer is clearly found in the context, respond accurately and concisely.
- If the context is insufficient or the answer is not present, say:
  "I could not find sufficient information in the provided documents."
- Do NOT make up information or rely on outside knowledge.
- When appropriate, summarize clearly instead of copying large chunks of text.
- Maintain a clear and structured answer.

Context:
{context}
"""


vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model_name=MODEL)
