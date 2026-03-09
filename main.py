from agent import RAGAgent


def main():

    agent = RAGAgent()

    print("RAG Agent started. Type 'exit' to stop.")

    while True:

        q = input("User: ")

        if q == "exit":
            break

        answer = agent.ask(q)

        print("Agent:", answer)


if __name__ == "__main__":
    main()