import logging
from tenacity import retry, wait_exponential, stop_after_attempt
import warnings

# suppress noisy warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*_target_device.*")

# silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
def retryable(func, *args, **kwargs):
    return func(*args, **kwargs)

def set_log_level(verbose: bool = False, debug: bool = False):
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(message)s")
    logger.setLevel(level)
