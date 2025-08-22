# --------------------------------------------------
# load_config.py - Updated with MongoDB support
# --------------------------------------------------
import json
import logging
from pathlib import Path
from typing import Any, Dict
from mongodb_store import init_mongodb_store, get_mongodb_store

# --------------------------------------------------------------------
# Logger
# --------------------------------------------------------------------
logger = logging.getLogger("discord-openai-proxy.config")
if not logger.handlers:
    hdlr = logging.StreamHandler()
    fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
    hdlr.setFormatter(logging.Formatter(fmt))
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

# --------------------------------------------------------------------
# Path constants (still needed for config.json)
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / "config.json"

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def _load_json_file(path: Path) -> Dict[str, Any]:
    """Return an empty dict if file missing; raise warning if JSON bad."""
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return {}
    try:
        content = path.read_text(encoding="utf-8")
        return json.loads(content) if content.strip() else {}
    except json.JSONDecodeError as exc:
        logger.error(f"Invalid JSON format in {path}:\n{exc}")
        return {}
    except Exception as exc:
        logger.exception(f"Error reading {path}: {exc}")
        return {}

def _int_or_default(val: Any, default: int, name: str) -> int:
    if val is None:
        logger.warning(f"{name} not defined in config; using default {default}")
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        logger.error(f"{name} must be an integer; using {default}")
        return default

# --------------------------------------------------------------------
# Đọc config.json
# --------------------------------------------------------------------
env_data: Dict[str, Any] = _load_json_file(ENV_FILE)

# --------------------------------------------------------------------
# Environment variables
# --------------------------------------------------------------------
DISCORD_TOKEN = env_data.get("DISCORD_TOKEN")
OPENAI_API_KEY = env_data.get("OPENAI_API_KEY")
OPENAI_API_BASE = env_data.get("OPENAI_API_BASE")
OPENAI_MODEL = env_data.get("OPENAI_MODEL")

# Gemini API configuration (NEW)
GEMINI_API_KEY = env_data.get("GEMINI_API_KEY")
OWNER_GEMINI_API_KEY = env_data.get("OWNER_GEMINI_API_KEY")  # New: Owner-specific key

# MongoDB configuration
MONGODB_CONNECTION_STRING = env_data.get("MONGODB_CONNECTION_STRING")
MONGODB_DATABASE_NAME = env_data.get("MONGODB_DATABASE_NAME", "discord_openai_proxy")
USE_MONGODB = env_data.get("USE_MONGODB", False)

# Tham số toàn cục
REQUEST_TIMEOUT = _int_or_default(env_data.get("REQUEST_TIMEOUT"), 100, "REQUEST_TIMEOUT")
MAX_MSG = _int_or_default(env_data.get("MAX_MSG"), 1900, "MAX_MSG")
MEMORY_MAX_PER_USER = _int_or_default(env_data.get("MEMORY_MAX_PER_USER"), 10, "MEMORY_MAX_PER_USER")
MEMORY_MAX_TOKENS = _int_or_default(env_data.get("MEMORY_MAX_TOKENS"), 2500, "MEMORY_MAX_TOKENS")

# --------------------------------------------------------------------
# Mandatory checks
# --------------------------------------------------------------------
if DISCORD_TOKEN is None or OPENAI_API_KEY is None:
    raise RuntimeError(
        "Both DISCORD_TOKEN and OPENAI_API_KEY must be defined in config.json."
    )

# MongoDB validation
if USE_MONGODB and not MONGODB_CONNECTION_STRING:
    raise RuntimeError(
        "USE_MONGODB is enabled but MONGODB_CONNECTION_STRING is not provided in config.json."
    )

# Supported models
SUPPORTED_MODELS = {"gpt-oss-20b", "gpt-oss-120b", "gpt-5", "o3-mini", "gpt-4.1"}
if OPENAI_MODEL and OPENAI_MODEL not in SUPPORTED_MODELS:
    logger.warning(f"MODEL {OPENAI_MODEL} not listed; should be monitored.")

# Gemini API validation (optional)
if GEMINI_API_KEY:
    logger.info("Gemini API key found - Gemini models will be available")
else:
    logger.warning("Gemini API key not found - Gemini models will not be available")

# --------------------------------------------------------------------
# Initialize MongoDB if enabled
# --------------------------------------------------------------------
_mongodb_initialized = False

def init_storage():
    """Initialize storage backend (MongoDB or file-based)"""
    global _mongodb_initialized
    
    if USE_MONGODB and not _mongodb_initialized:
        try:
            init_mongodb_store(MONGODB_CONNECTION_STRING, MONGODB_DATABASE_NAME)
            logger.info(f"MongoDB initialized: {MONGODB_DATABASE_NAME}")
            _mongodb_initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {e}")
            raise RuntimeError(f"MongoDB initialization failed: {e}")
    elif not USE_MONGODB:
        logger.info("Using file-based storage (legacy mode)")

def get_storage_type() -> str:
    """Get current storage type"""
    return "mongodb" if USE_MONGODB else "file"

# --------------------------------------------------------------------
# System prompt loader (DEPRECATED - kept for backward compatibility)
# --------------------------------------------------------------------
def load_system_prompt() -> Dict[str, str]:
    """
    DEPRECATED: System prompts are now managed per-user.
    This function is kept for backward compatibility but will return empty.
    """
    logger.warning("load_system_prompt() is deprecated. System prompts are now managed per-user.")
    return {"role": "system", "content": ""}