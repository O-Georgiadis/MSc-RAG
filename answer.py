from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


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


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    return retriever.invoke(question)


def combined_questions(question, history: list[dict] = None) -> str:
    """
    Combine all the user's messages into a single string.
    """
    if history is None:
        history = []

    # Normalise the current question to a string
    if isinstance(question, list):
        question = "\n".join(str(q) for q in question)
    else:
        question = str(question)

    user_contents: list[str] = []
    for m in history:
        # Handle our own format: {"role": "...", "content": ...}
        if isinstance(m, dict):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, list):
                    user_contents.append("\n".join(str(c) for c in content))
                else:
                    user_contents.append(str(content))
        # Handle Gradio Chatbot default format: [user_message, assistant_message]
        elif isinstance(m, (list, tuple)) and len(m) >=1:
            user_contents.append(str(m[0]))

    prior = "\n".join(user_contents)
    if prior:
        return prior + "\n" + question
    return question


def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG. Return the answer and the context documents.
    """
    combined = combined_questions(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)

    # Build LangChain chat history explicitly to handle both our dict format
    # and Gradio's default [user, assistant] pairs, avoiding type errors.
    lc_history: list[SystemMessage | HumanMessage | AIMessage] = []
    for m in history:
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content", "")
            if isinstance(content, list):
                content = "\n".join(str(c) for c in content)
            else:
                content = str(content)

            if role == "user":
                lc_history.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_history.append(AIMessage(content=content))
        # If history is in Gradio [user, assistant] format
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            user_msg, assistant_msg = m
            lc_history.append(HumanMessage(content=str(user_msg)))
            lc_history.append(AIMessage(content=str(assistant_msg)))

    message = [SystemMessage(content=system_prompt), *lc_history, HumanMessage(content=question)]
    response = llm.invoke(message)
    return response.content, docs


