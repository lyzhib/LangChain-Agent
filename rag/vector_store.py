from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_vector_store():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    with open("data/knowledge.txt", "r", encoding="utf-8") as f:
        text = f.read()

    docs = text.split("\n\n")

    vectorstore = FAISS.from_texts(
        docs,
        embeddings
    )

    return vectorstore
