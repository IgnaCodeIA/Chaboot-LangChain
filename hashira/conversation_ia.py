from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from rich.console import Console

from utils import (DocsJSONLLoader, get_file_path, get_openai_api_key,
                   get_query_from_user)

console = Console()
create_chroma_db = False
chat_type = "memory_chat"


def load_documents(file_path: str) -> list[Document]:
    loader = DocsJSONLLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=6000, length_function=len, chunk_overlap=500  # 10% overlap
    )

    return text_splitter.split_documents(documents)


def get_chroma_db(embedding, documents, path):
    if create_chroma_db:
        console.print("[bold green]Creando Chroma DB...[/bold green]")
        return Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            persist_directory=path,
        )
    else:
        console.print(
            f"[bold green]Cargando Chroma DB existente cantidad de documentos {len(documents)}...[/bold green]"
        )
        return Chroma(persist_directory=path, embedding_function=embedding)


def process_qa_query(query, retriver, llm):

    qa_chain = RetrievalQA.from_llm(llm=llm, retriever=retriver)
    console.print("[yellow]La IA esta pensando...[/yellow]")

    return qa_chain.run(query)


def process_memory_query(query, retriver, llm, chat_history):
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriver, verbose=True
    )
    console.print("[yellow]La IA esta pensando...[/yellow]")
    console.print(f"[blue]Historial de la conversación:[/blue] {chat_history}")
    result = conversation({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))
    return result["answer"]


def run_conversation(vector_store, chat_type, llm):
    console.print(
        "\n[blue]IA:[/blue] Hola 🚀! Que quieres preguntarme sobre Transformers e inteligencia artificial 🤖 en general?"
    )
    if chat_type == "qa":
        console.print(
            "\n[green]Estas utilizando el chatbot 🤖 en modo de preguntas y respuestas. Este chatbot 🤖 genera respuestas basandose puramente en la consulta actual sin considerar el historial de la conversación.[/green]"
        )
    elif chat_type == "memory_chat":
        console.print(
            "\n[green]Estás utilizando el chatbot 🤖 en modo de memoria. Este chatbot genera respuestas basándose en el historial de la conversación y en la consulta actual.[/green]"
        )

    retriver = vector_store.as_retriever(
        search_kwargs={
            "k": 2,
        }
    )
    chat_history = []
    while True:
        console.print("\n[blue]Tú:[/blue]")
        query = get_query_from_user()
        if query.lower() == "salir":
            break
        if chat_type == "qa":
            response = process_qa_query(query=query, retriver=retriver, llm=llm)
            console.print(f"\n[blue]IA:[/blue] {response}")
        elif chat_type == "memory_chat":
            response = process_memory_query(
                query=query, retriver=retriver, llm=llm, chat_history=chat_history
            )
            console.print(f"\n[blue]IA:[/blue] {response}")


def main():
    documents = load_documents(get_file_path())
    get_openai_api_key()
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    console.print(f"[bold green]Documentos {len(documents)} cargados.[/bold green]")
    vector_store = get_chroma_db(embedding, documents, "chroma_docs")

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=1000,
    )
    run_conversation(vector_store, chat_type, llm)


if __name__ == "__main__":
    main()
