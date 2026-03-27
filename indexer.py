from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import CHROMA_PERSIST_DIR, CHUNK_SIZE, CHUNK_OVERLAP, DEFAULT_EMBEDDER, DEFAULT_EMBED_MODEL
from utils import logger

def _get_instructor_embeddings(model_name: str = "hkunlp/instructor-base"):
    from langchain_community.embeddings import HuggingFaceInstructEmbeddings
    return HuggingFaceInstructEmbeddings(model_name=model_name)

def _get_hf_minilm(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)

def _get_openai_embeddings(model_name: Optional[str] = None):
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model_name) if model_name else OpenAIEmbeddings()

def _get_cohere_embeddings(model_name: Optional[str] = None):
    from langchain_cohere import CohereEmbeddings
    return CohereEmbeddings(model=model_name) if model_name else CohereEmbeddings()

def build_vectorstore(
    docs: List[Document],
    persist_dir: str = CHROMA_PERSIST_DIR,
    embedder: str = DEFAULT_EMBEDDER,
    embed_model: Optional[str] = None
):
    """
    Build Chroma vectorstore from documents.

    embedder: 'huggingface' (default), 'instructor', 'openai', 'cohere'
    embed_model: optional model name
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    splits = splitter.split_documents(docs)

    embedder = embedder.lower()
    try:
        if embedder == "instructor":
            emb = _get_instructor_embeddings(embed_model or DEFAULT_EMBED_MODEL)
        elif embedder in ("hf", "huggingface", "minilm"):
            emb = _get_hf_minilm(embed_model or DEFAULT_EMBED_MODEL)
        elif embedder == "openai":
            emb = _get_openai_embeddings(embed_model)
        elif embedder == "cohere":
            emb = _get_cohere_embeddings(embed_model)
        else:
            logger.warning(
                "Unknown embedder '%s', defaulting to HuggingFace MiniLM", embedder
            )
            emb = _get_hf_minilm(embed_model or DEFAULT_EMBED_MODEL)
    except Exception as e:
        logger.exception("Failed to initialize embedder '%s': %s", embedder, e)
        raise

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=emb,
        persist_directory=persist_dir
    )
    vectorstore.persist()
    logger.info(
        "Vectorstore built and persisted at '%s' using embedder '%s'", persist_dir, embedder
    )
    return vectorstore

def load_vectorstore(
    persist_dir: str = CHROMA_PERSIST_DIR,
    embedder: str = DEFAULT_EMBEDDER,
    embed_model: Optional[str] = DEFAULT_EMBED_MODEL,
    init_embedder: bool = False, 
):
    emb = None
    if init_embedder:
        try:
            if embedder == "instructor":
                emb = _get_instructor_embeddings(embed_model or "hkunlp/instructor-base")
            elif embedder in ("hf", "huggingface", "minilm"):
                emb = _get_hf_minilm(embed_model or "sentence-transformers/all-MiniLM-L6-v2")
            elif embedder == "openai":
                emb = _get_openai_embeddings(embed_model)
            elif embedder == "cohere":
                emb = _get_cohere_embeddings(embed_model)
        except Exception:
            logger.warning("Could not initialize embedder; will open Chroma without embedding function")
            emb = None

    try:
        if emb is not None:
            return Chroma(persist_directory=persist_dir, embedding_function=emb)
        else:
            return Chroma(persist_directory=persist_dir)
    except Exception as e:
        logger.exception("Failed to load Chroma DB: %s", e)
        raise