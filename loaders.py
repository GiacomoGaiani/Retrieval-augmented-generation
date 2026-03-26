from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_core.documents import Document
import os
from utils import logger


def load_blog(path_or_url: str):
    """Load a document or multiple documents from URL, file, or directory."""
    docs = []

    # URL
    if path_or_url.startswith("http"):
        logger.info("→ Loading from URL: %s", path_or_url)
        loader = WebBaseLoader(path_or_url)
        return loader.load()

    # Directory: recursively load .pdf and .txt files
    if os.path.isdir(path_or_url):
        logger.info("→ Loading all supported documents from directory: %s", path_or_url)
        for root, _, files in os.walk(path_or_url):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    if fname.lower().endswith(".pdf"):
                        docs.extend(load_pdf(fpath))
                    elif fname.lower().endswith(".txt"):
                        docs.extend(load_text(fpath))
                    else:
                        logger.debug("Skipping unsupported file: %s", fpath)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", fpath, e)
        return docs

    # Single file (PDF or TXT)
    if os.path.isfile(path_or_url):
        if path_or_url.lower().endswith(".pdf"):
            return load_pdf(path_or_url)
        elif path_or_url.lower().endswith(".txt"):
            return load_text(path_or_url)
        else:
            raise ValueError(f"Unsupported file type: {path_or_url}")

    raise ValueError(f"Path or URL not found: {path_or_url}")


def load_pdf(path):
    """Load a single PDF."""
    logger.info("→ Loading PDF: %s", path)
    loader = PyPDFLoader(path)
    return loader.load()


def load_text(path):
    """Load a single text file."""
    logger.info("→ Loading text file: %s", path)
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()
