from rag.rag_chain import create_rag_chain


class RAGAgent:

    def __init__(self):

        self.chain = create_rag_chain()

    def ask(self, question):

        return self.chain.invoke(question)