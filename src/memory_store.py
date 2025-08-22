import json
import tiktoken
import logging
from pathlib import Path
from collections import deque
from typing import Dict, List, Union, TypedDict
import load_config

logger = logging.getLogger("discord-openai-proxy.memory_store")

# Remove auto-creation and use parent directory directly
CONFIG_DIR = Path(__file__).parent.parent / "config"

TOKENIZER = tiktoken.encoding_for_model("gpt-4")

class Msg(TypedDict):
    role: str
    content: str

class MemoryStore:
    def __init__(self, path: Union[Path, str] = None):
        self.use_mongodb = load_config.USE_MONGODB
        
        if self.use_mongodb:
            # MongoDB mode
            from mongodb_store import get_mongodb_store
            self.mongo_store = get_mongodb_store()
            logger.info("MemoryStore initialized with MongoDB")
        else:
            # File mode (legacy)
            self.path = Path(path) if path else (CONFIG_DIR / 'memory.json')
            
            # {user_id: deque([msg, ...])}
            self._cache: Dict[int, deque[Msg]] = {}
            self._token_cnt: Dict[int, int] = {}
            self._load()
            logger.info("MemoryStore initialized with file storage")

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load từ file (file mode only)"""
        if self.use_mongodb:
            return
            
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for k, v in data.items():
                uid = int(k)
                # Build deque with tokens counted once
                d = deque()
                total_tokens = 0
                for m in v:
                    d.append(m)
                    total_tokens += len(TOKENIZER.encode(m["content"]))
                self._cache[uid] = d
                self._token_cnt[uid] = total_tokens
        except Exception:  # pragma: no cover
            logger.exception("Error loading memory from file")
            self._cache, self._token_cnt = {}, {}

    def _save(self) -> None:
        """Save to file (file mode only)"""
        if self.use_mongodb:
            return
            
        try:
            # Remove auto-creation of directory
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(
                {str(k): list(v) for k, v in self._cache.items()},
                indent=2,
                ensure_ascii=False
            ), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:  # pragma: no cover
            logger.exception("Error saving memory to file")

    # ------------------------------------------------------------------
    def get_user_messages(self, user_id: int) -> List[Msg]:
        """Get user's conversation history"""
        if self.use_mongodb:
            return self.mongo_store.get_user_messages(user_id)
        else:
            return list(self._cache.get(user_id, []))

    def add_message(self, user_id: int, msg: Msg) -> None:
        """Add message to user's conversation history"""
        if self.use_mongodb:
            max_messages = getattr(load_config, 'MEMORY_MAX_PER_USER', 50)
            max_tokens = getattr(load_config, 'MEMORY_MAX_TOKENS', 2000)
            self.mongo_store.add_message(user_id, msg, max_messages, max_tokens)
        else:
            self._cache.setdefault(user_id, deque()).append(msg)
            
            self._token_cnt[user_id] = self._token_cnt.get(user_id, 0) \
                                         + len(TOKENIZER.encode(msg["content"]))
            
            self._prune(user_id)
            self._save()

    def clear_user(self, user_id: int) -> None:
        """Clear user's conversation history"""
        if self.use_mongodb:
            self.mongo_store.clear_user_memory(user_id)
        else:
            self._cache.pop(user_id, None)
            self._token_cnt.pop(user_id, None)
            self._save()

    # ------------------------------------------------------------------
    def _prune(self, user_id: int) -> None:
        """Prune messages (file mode only)"""
        if self.use_mongodb:
            return
            
        d = self._cache[user_id]
        token_cnt = self._token_cnt.get(user_id, 0)
        
        max_messages = getattr(load_config, 'MEMORY_MAX_PER_USER', 50)
        max_tokens = getattr(load_config, 'MEMORY_MAX_TOKENS', 2000)

        while len(d) > max_messages:
            removed = d.popleft()
            token_cnt -= len(TOKENIZER.encode(removed["content"]))
            
        while token_cnt > max_tokens and d:
            removed = d.popleft()
            token_cnt -= len(TOKENIZER.encode(removed["content"]))

        self._token_cnt[user_id] = token_cnt