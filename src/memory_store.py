# Fixed memory_store.py - Replace the entire file

import json
import tiktoken
import logging
from pathlib import Path
from collections import deque
from typing import Dict, List, Union, TypedDict
import load_config

logger = logging.getLogger("memory_store")

# Configuration
CONFIG_DIR = Path(__file__).parent.parent / "config"

try:
    TOKENIZER = tiktoken.encoding_for_model("gpt-4")
except:
    TOKENIZER = tiktoken.get_encoding("cl100k_base")

class Msg(TypedDict):
    role: str
    content: str

class MemoryStore:
    def __init__(self, path: Union[Path, str] = None):
        self.use_mongodb = getattr(load_config, 'USE_MONGODB', False)
        
        if self.use_mongodb:
            # MongoDB mode - delegate to MongoDB store
            try:
                from mongodb_store import get_mongodb_store
                self.mongo_store = get_mongodb_store()
                logger.info("MemoryStore initialized with MongoDB backend")
            except Exception as e:
                logger.error(f"Failed to initialize MongoDB backend: {e}")
                logger.info("Falling back to file-based storage")
                self.use_mongodb = False
                self._init_file_storage(path)
        else:
            # File mode (legacy)
            self._init_file_storage(path)
            
    def _init_file_storage(self, path):
        """Initialize file-based storage"""
        self.path = Path(path) if path else (CONFIG_DIR / 'memory.json')
        self._cache: Dict[int, deque[Msg]] = {}
        self._token_cnt: Dict[int, int] = {}
        self._load()
        logger.info("MemoryStore initialized with file storage")

    def _load(self) -> None:
        """Load from file (file mode only)"""
        if self.use_mongodb:
            return
            
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for k, v in data.items():
                uid = int(k)
                d = deque()
                total_tokens = 0
                for m in v:
                    d.append(m)
                    total_tokens += len(TOKENIZER.encode(m["content"]))
                self._cache[uid] = d
                self._token_cnt[uid] = total_tokens
        except Exception:
            logger.exception("Error loading memory from file")
            self._cache, self._token_cnt = {}, {}

    def _save(self) -> None:
        """Save to file (file mode only)"""
        if self.use_mongodb:
            return
            
        try:
            # Ensure directory exists
            self.path.parent.mkdir(parents=True, exist_ok=True)
            
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(
                {str(k): list(v) for k, v in self._cache.items()},
                indent=2,
                ensure_ascii=False
            ), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:
            logger.exception("Error saving memory to file")

    def get_user_messages(self, user_id: int) -> List[Msg]:
        """Get user's conversation history"""
        if self.use_mongodb:
            try:
                messages = self.mongo_store.get_user_messages(user_id)
                logger.debug(f"Retrieved {len(messages)} messages from MongoDB for user {user_id}")
                return messages
            except Exception as e:
                logger.exception(f"Error getting messages from MongoDB for user {user_id}: {e}")
                return []
        else:
            return list(self._cache.get(user_id, []))

    def add_message(self, user_id: int, msg: Msg) -> bool:
        """Add message to user's conversation history"""
        if self.use_mongodb:
            try:
                max_messages = getattr(load_config, 'MEMORY_MAX_PER_USER', 25)
                max_tokens = getattr(load_config, 'MEMORY_MAX_TOKENS', 4000)
                
                success = self.mongo_store.add_message(user_id, msg, max_messages, max_tokens)
                if success:
                    logger.debug(f"Successfully added message to MongoDB for user {user_id}")
                else:
                    logger.error(f"Failed to add message to MongoDB for user {user_id}")
                return success
            except Exception as e:
                logger.exception(f"Error adding message to MongoDB for user {user_id}: {e}")
                return False
        else:
            # File-based storage
            self._cache.setdefault(user_id, deque()).append(msg)
            
            self._token_cnt[user_id] = self._token_cnt.get(user_id, 0) + len(TOKENIZER.encode(msg["content"]))
            
            self._prune(user_id)
            self._save()
            return True

    def clear_user(self, user_id: int) -> bool:
        """Clear user's conversation history"""
        if self.use_mongodb:
            try:
                success = self.mongo_store.clear_user_memory(user_id)
                if success:
                    logger.info(f"Successfully cleared memory for user {user_id} from MongoDB")
                else:
                    logger.warning(f"Failed to clear memory for user {user_id} from MongoDB")
                return success
            except Exception as e:
                logger.exception(f"Error clearing memory from MongoDB for user {user_id}: {e}")
                return False
        else:
            # File-based storage
            self._cache.pop(user_id, None)
            self._token_cnt.pop(user_id, None)
            self._save()
            return True

    def remove_last_message(self, user_id: int) -> bool:
        """Remove the last message from user's conversation history"""
        if self.use_mongodb:
            try:
                success = self.mongo_store.remove_last_message(user_id)
                if success:
                    logger.info(f"Successfully removed last message for user {user_id} from MongoDB")
                else:
                    logger.warning(f"Failed to remove last message for user {user_id} from MongoDB")
                return success
            except Exception as e:
                logger.exception(f"Error removing last message from MongoDB for user {user_id}: {e}")
                return False
        else:
            # File-based storage
            if user_id not in self._cache:
                return False
            
            d = self._cache[user_id]
            if not d:
                return False
                
            removed = d.pop()
            self._token_cnt[user_id] = self._token_cnt.get(user_id, 0) - len(TOKENIZER.encode(removed["content"]))
            
            if self._token_cnt[user_id] < 0:
                self._token_cnt[user_id] = 0
                
            self._save()
            return True

    def _prune(self, user_id: int) -> None:
        """Prune messages (file mode only)"""
        if self.use_mongodb:
            return
            
        d = self._cache[user_id]
        token_cnt = self._token_cnt.get(user_id, 0)
        
        max_messages = getattr(load_config, 'MEMORY_MAX_PER_USER', 50)
        max_tokens = getattr(load_config, 'MEMORY_MAX_TOKENS', 4000)

        while len(d) > max_messages:
            removed = d.popleft()
            token_cnt -= len(TOKENIZER.encode(removed["content"]))
            
        while token_cnt > max_tokens and d:
            removed = d.popleft()
            token_cnt -= len(TOKENIZER.encode(removed["content"]))

        self._token_cnt[user_id] = max(0, token_cnt)  # Ensure non-negative

    def get_message_count(self, user_id: int) -> int:
        """Get the number of messages for a user"""
        if self.use_mongodb:
            messages = self.get_user_messages(user_id)
            return len(messages)
        else:
            return len(self._cache.get(user_id, []))

    def get_token_count(self, user_id: int) -> int:
        """Get approximate token count for user's messages"""
        if self.use_mongodb:
            messages = self.get_user_messages(user_id)
            return sum(len(TOKENIZER.encode(msg.get("content", ""))) for msg in messages)
        else:
            return self._token_cnt.get(user_id, 0)