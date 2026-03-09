from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from rag.vector_store import create_vector_store
from llm.gigachat_studio import GigaChatStudioLLM


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain():

    llm = GigaChatStudioLLM()

    vectorstore = create_vector_store()

    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        """
Answer using the context.

Context:
{context}

Question:
{question}
"""
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
