#!/usr/bin/env python3
# coding: utf-8
# user_config.py - User configuration management with MongoDB support

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Set
import load_config

logger = logging.getLogger("user_config")

# User config file path (fallback for file mode)
BASE_DIR = Path(__file__).resolve().parent.parent
CONF_DIR = BASE_DIR / "config"
USER_CONFIG_FILE = CONF_DIR / "user_config.json"

# Supported models (FALLBACK - will be fetched from database)
FALLBACK_SUPPORTED_MODELS = {
    "o3-mini",
    "gpt-4.1",
    "gpt-5",
    "gpt-oss-20b",
    "gpt-oss-120b", 
}

# Default system prompt
DEFAULT_SYSTEM_PROMPT = (
    "Tên của bạn là Ryuuko (nữ), nói tiếng việt"
)

# Default model
DEFAULT_MODEL = "gemini-2.5-flash"

class UserConfigManager:
    def __init__(self):
        self.use_mongodb = load_config.USE_MONGODB
        
        if self.use_mongodb:
            # MongoDB mode
            from mongodb_store import get_mongodb_store
            self.mongo_store = get_mongodb_store()
            logger.info("UserConfigManager initialized with MongoDB")
        else:
            # File mode (legacy)
            self.config_file = USER_CONFIG_FILE
            self._config_cache: Dict[str, Dict[str, Any]] = {}
            self._load_config()
            logger.info("UserConfigManager initialized with file storage")
    
    def _load_config(self) -> None:
        """Load configuration from JSON file (file mode only)"""
        if self.use_mongodb:
            return
            
        if not self.config_file.exists():
            self._config_cache = {}
            return
            
        try:
            content = self.config_file.read_text(encoding="utf-8")
            self._config_cache = json.loads(content) if content.strip() else {}
            logger.info(f"Loaded config for {len(self._config_cache)} users")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in user_config.json: {e}")
            self._config_cache = {}
        except Exception as e:
            logger.exception(f"Error loading user_config.json: {e}")
            self._config_cache = {}
    
    def _save_config(self) -> None:
        """Save configuration to JSON file (file mode only)"""
        if self.use_mongodb:
            return
            
        try:
            # Remove auto-creation of directory
            # Ghi vào file tạm trước
            tmp_file = self.config_file.with_suffix('.tmp')
            tmp_file.write_text(
                json.dumps(self._config_cache, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            
            # Sau đó move sang file chính
            tmp_file.replace(self.config_file)
            logger.debug(f"Saved config for {len(self._config_cache)} users")
            
        except Exception as e:
            logger.exception(f"Error saving user_config.json: {e}")
    
    def get_supported_models(self) -> Set[str]:
        """Get set of all supported model names"""
        if self.use_mongodb:
            # Get both regular and profile models
            models = self.mongo_store.get_supported_models()
            profiles = {p['name'] for p in self.mongo_store.list_profile_models()}
            return models.union(profiles)
        else:
            # Legacy file mode
            return FALLBACK_SUPPORTED_MODELS
    
    def add_supported_model(self, model_name: str, credit_cost: int = 1, access_level: int = 0) -> tuple[bool, str]:
        """
        Add a new supported model with credit cost and access level
        Returns: (success: bool, message: str)
        """
        if not self.use_mongodb:
            return False, "Model management requires MongoDB mode"
        
        return self.mongo_store.add_supported_model(model_name, credit_cost, access_level)
    
    def remove_supported_model(self, model_name: str) -> tuple[bool, str]:
        """
        Remove a supported model (MongoDB only)
        Returns: (success: bool, message: str)
        """
        if not self.use_mongodb:
            return False, "Model management requires MongoDB mode"
        
        return self.mongo_store.remove_supported_model(model_name)
    
    def list_all_models_detailed(self) -> list:
        """Get detailed list of all models (MongoDB only)"""
        if not self.use_mongodb:
            return []
        
        return self.mongo_store.list_all_models()
    
    def get_user_config(self, user_id: int) -> Dict[str, Any]:
        """Get user configuration, create default if not exists"""
        if self.use_mongodb:
            return self.mongo_store.get_user_config(user_id)
        else:
            # File mode
            user_key = str(user_id)
            if user_key not in self._config_cache:
                self._config_cache[user_key] = {
                    "model": DEFAULT_MODEL,
                    "system_prompt": DEFAULT_SYSTEM_PROMPT
                }
                self._save_config()
            return self._config_cache[user_key]
    
    def set_user_model(self, user_id: int, model: str) -> tuple[bool, str]:
        """
        Set user's model
        Returns: (success: bool, message: str)
        """
        supported_models = self.get_supported_models()
        if model not in supported_models:
            supported_list = ", ".join(sorted(supported_models))
            return False, f"Model '{model}' not supported. Available models: {supported_list}"
        
        if self.use_mongodb:
            success = self.mongo_store.set_user_config(user_id, model=model)
            if success:
                return True, f"Model set to '{model}'"
            else:
                return False, "Error saving configuration to database"
        else:
            # File mode
            user_config = self.get_user_config(user_id)
            user_config["model"] = model
            self._save_config()
            return True, f"Model set to '{model}'"
    
    def set_user_system_prompt(self, user_id: int, prompt: str) -> tuple[bool, str]:
        """
        Set user's system prompt
        Returns: (success: bool, message: str)
        """
        if not prompt.strip():
            return False, "System prompt cannot be empty"
        
        if len(prompt) > 10000:  # Length limit
            return False, "System prompt too long (max 10,000 characters)"
        
        if self.use_mongodb:
            success = self.mongo_store.set_user_config(user_id, system_prompt=prompt.strip())
            if success:
                return True, "System prompt updated"
            else:
                return False, "Error saving configuration to database"
        else:
            # File mode
            user_config = self.get_user_config(user_id)
            user_config["system_prompt"] = prompt.strip()
            self._save_config()
            return True, "System prompt updated"
    
    def get_user_model(self, user_id: int) -> str:
        """Get user's current model"""
        if self.use_mongodb:
            return self.mongo_store.get_user_model(user_id)
        else:
            return self.get_user_config(user_id)["model"]
    
    def get_user_system_prompt(self, user_id: int) -> str:
        """Get user's current system prompt"""
        if self.use_mongodb:
            return self.mongo_store.get_user_system_prompt(user_id)
        else:
            return self.get_user_config(user_id)["system_prompt"]
    
    def get_user_system_message(self, user_id: int) -> Dict[str, str]:
        """Get system message in OpenAI format"""
        if self.use_mongodb:
            return self.mongo_store.get_user_system_message(user_id)
        else:
            return {
                "role": "system",
                "content": self.get_user_system_prompt(user_id)
            }
    
    def reset_user_config(self, user_id: int) -> str:
        """Reset user config to defaults"""
        if self.use_mongodb:
            # Reset config in MongoDB
            success = self.mongo_store.set_user_config(user_id, model=DEFAULT_MODEL, system_prompt=DEFAULT_SYSTEM_PROMPT)
            if success:
                return "Configuration reset to defaults"
            else:
                return "Error resetting configuration"
        else:
            # File mode
            user_key = str(user_id)
            if user_key in self._config_cache:
                del self._config_cache[user_key]
                self._save_config()
                return "Configuration reset to defaults"
            return "No configuration to reset"

# Singleton instance
_user_config_manager = None

def get_user_config_manager() -> UserConfigManager:
    """Get UserConfigManager instance (singleton pattern)"""
    global _user_config_manager
    if _user_config_manager is None:
        _user_config_manager = UserConfigManager()
    return _user_config_manager

# Legacy functions for backward compatibility (DEPRECATED)
def get_supported_models() -> Set[str]:
    """DEPRECATED: Use get_user_config_manager().get_supported_models() instead"""
    logger.warning("get_supported_models() is deprecated. Use get_user_config_manager().get_supported_models()")
    return get_user_config_manager().get_supported_models()

# Make supported models available as module variable for backward compatibility
SUPPORTED_MODELS = FALLBACK_SUPPORTED_MODELS  # This will be updated dynamically