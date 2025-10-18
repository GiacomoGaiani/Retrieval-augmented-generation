from dotenv import load_dotenv
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)

load_dotenv()

OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY: Optional[str] = os.getenv("COHERE_API_KEY")
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "3000"))

DEFAULT_EMBEDDER: str = os.getenv("DEFAULT_EMBEDDER", "instructor")  # or "huggingface"
DEFAULT_EMBED_MODEL: Optional[str] = os.getenv("DEFAULT_EMBED_MODEL", "hkunlp/instructor-base")

LLM_MODEL = os.getenv("LLM_MODEL", "mistral")