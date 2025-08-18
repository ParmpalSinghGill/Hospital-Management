import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


def main() -> None:
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY not set. Put it in your environment or in a .env file.")

    # Minimal LangChain pipeline: Prompt -> Groq LLM -> Output
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise assistant."),
        ("human", "{input}")
    ])

    llm = ChatGroq(model="llama3-8b-8192", temperature=0)

    chain = prompt | llm

    result = chain.invoke({
        "input": "Say 'Hello from Groq + LangChain!' in one short sentence."
    })

    # result is a ChatMessage; print the content
    print(result.content)


if __name__ == "__main__":
    main()
