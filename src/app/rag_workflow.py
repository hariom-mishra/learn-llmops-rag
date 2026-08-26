from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_chroma import Chroma

from langgraph.graph import StateGraph, START, END
from langfuse import get_client

from src.configs.config import params_config

# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# Models
# --------------------------------------------------

llm = ChatOpenAI(
    model=params_config.rag_app.llm
)

embedding_model = OpenAIEmbeddings(
    model=params_config.rag_app.embedding_model,
    dimensions=params_config.rag_app.embedding_dimensions
)


# --------------------------------------------------
# Paths and constants
# --------------------------------------------------

_ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DOCUMENTS_PATH = _ROOT_DIR / "data" / "processed"
VECTOR_DB_PATH = _ROOT_DIR / "saved_embeddings"
COLLECTION_NAME = params_config.rag_app.collection_name


# --------------------------------------------------
# State
# --------------------------------------------------

class RAGState(TypedDict):
    query: str
    prompt: ChatPromptTemplate
    retrieved_documents: list[Document]
    response: str
    context: str


# --------------------------------------------------
# Create / Load Vector Store
# --------------------------------------------------

def get_vector_store() -> Chroma:
    """
    Connect to the existing Chroma vector database.
    """

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=VECTOR_DB_PATH.as_posix()
    )


# --------------------------------------------------
# Index documents only if DB is empty
# --------------------------------------------------

def initialize_vector_store():
    """
    If embeddings already exist, do nothing.

    If the database is empty, load documents,
    split them, create embeddings, and store them.
    """

    vector_store = get_vector_store()

    # Check how many documents already exist
    existing_documents = vector_store._collection.count()

    # Database already has embeddings
    if existing_documents > 0:
        print(
            f"Vector database already exists "
            f"with {existing_documents} documents."
        )
        print("Skipping indexing.")
        return

    print("No existing embeddings found.")
    print("Starting indexing...")

    # ----------------------------------------------
    # Load documents
    # ----------------------------------------------

    text_loader = DirectoryLoader(
        DOCUMENTS_PATH.as_posix(),
        loader_cls=TextLoader
    )

    loaded_docs = text_loader.load()

    print(f"Loaded {len(loaded_docs)} documents.")

    # ----------------------------------------------
    # Split documents
    # ----------------------------------------------

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=params_config.rag_app.tokenizer_encoding,
        chunk_size=params_config.rag_app.chunk_size,
        chunk_overlap=params_config.rag_app.chunk_overlap
    )

    docs = splitter.split_documents(loaded_docs)

    print(f"Created {len(docs)} chunks.")

    # ----------------------------------------------
    # Create embeddings and store them
    # ----------------------------------------------

    vector_store.add_documents(docs)

    print("Embeddings created and stored successfully.")


# --------------------------------------------------
# Retriever Node
# --------------------------------------------------

def retriever(state: RAGState) -> dict:

    vector_store = get_vector_store()

    db_retriever = vector_store.as_retriever(
        search_type=params_config.rag_app.search_type,
        search_kwargs={"k": params_config.rag_app.k}
    )

    query = state["query"]

    retrieved_docs = db_retriever.invoke(query)

    context = "\n\n".join(
        document.page_content
        for document in retrieved_docs
    )

    return {
        "retrieved_documents": retrieved_docs,
        "context": context
    }


# --------------------------------------------------
# Augmentation Node
# --------------------------------------------------

def augmentation(state: RAGState) -> dict:
    langfuse = get_client()

    system_prompt = langfuse.get_prompt(
        name="rag_app_system_prompt",
        type="text",
        label=params_config.rag_app.prompt_label
    )
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            system_prompt.prompt
            #             """
            # You are a helpful assistant.

            # Answer the user query based only on the given context.

            # If the answer is not available in the context, say:
            # "I don't know."

            # Do not add any preamble to the response.
            # """
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{query}"
        )
    ])

    return {
        "prompt": prompt
    }


# --------------------------------------------------
# Generation Node
# --------------------------------------------------

def generation(state: RAGState) -> dict:

    query = state["query"]
    context = state["context"]
    prompt = state["prompt"]

    parser = StrOutputParser()

    chain = prompt | llm | parser

    response = chain.invoke({
        "query": query,
        "context": context
    })

    return {
        "response": response
    }


# --------------------------------------------------
# Initialize database
# --------------------------------------------------

# This runs when the application starts.
# First time -> indexes documents.
# Later runs -> detects existing documents and skips indexing.
initialize_vector_store()


# --------------------------------------------------
# Build LangGraph
# --------------------------------------------------

state_builder = StateGraph(RAGState)

state_builder.add_node(
    "retriever",
    retriever
)

state_builder.add_node(
    "augmentation",
    augmentation
)

state_builder.add_node(
    "generation",
    generation
)


# --------------------------------------------------
# Graph flow
# --------------------------------------------------

state_builder.add_edge(
    START,
    "retriever"
)

state_builder.add_edge(
    "retriever",
    "augmentation"
)

state_builder.add_edge(
    "augmentation",
    "generation"
)

state_builder.add_edge(
    "generation",
    END
)


# --------------------------------------------------
# Compile graph
# --------------------------------------------------

graph = state_builder.compile()