#!/usr/bin/env python3
# coding: utf-8
# mongodb_store.py - MongoDB storage for user configs, memory, authorized users, and supported models

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import tiktoken

logger = logging.getLogger("discord-openai-proxy.mongodb_store")

class MongoDBStore:
    """MongoDB storage manager for Discord OpenAI proxy"""
    
    def __init__(self, connection_string: str, database_name: str = "discord_openai_proxy"):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client: Optional[MongoClient] = None
        self.db = None
        self.tokenizer = tiktoken.encoding_for_model("gpt-oss-120b")
        
        # Collection names
        self.COLLECTIONS = {
            'user_config': 'user_configs',
            'memory': 'user_memory', 
            'authorized': 'authorized_users',
            'models': 'supported_models'  # NEW: Collection for supported models
        }
        
        self._connect()
        self._initialize_default_models()
    
    def _connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            
            # Create indexes for better performance
            self._create_indexes()
            
            logger.info(f"Successfully connected to MongoDB: {self.database_name}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error connecting to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create necessary indexes"""
        try:
            # User config indexes
            self.db[self.COLLECTIONS['user_config']].create_index("user_id", unique=True)
            
            # Memory indexes
            self.db[self.COLLECTIONS['memory']].create_index("user_id", unique=True)
            
            # Authorized users indexes
            self.db[self.COLLECTIONS['authorized']].create_index("user_id", unique=True)
            
            # Supported models indexes
            self.db[self.COLLECTIONS['models']].create_index("model_name", unique=True)
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.exception(f"Error creating indexes: {e}")
    
    def _initialize_default_models(self):
        """Initialize default supported models if collection is empty"""
        try:
            # Check if models collection exists and has data
            count = self.db[self.COLLECTIONS['models']].count_documents({})
            if count == 0:
                default_models = [
                    {"model_name": "gemini-2.5-flash", "created_at": datetime.utcnow(), "is_default": True, "credit_cost": 10, "access_level": 0},
                    {"model_name": "gemini-2.5-pro", "created_at": datetime.utcnow(), "is_default": True, "credit_cost": 50, "access_level": 0},
                    {"model_name": "gpt-3.5-turbo", "created_at": datetime.utcnow(), "is_default": True, "credit_cost": 200, "access_level": 0},
                    {"model_name": "gpt-5", "created_at": datetime.utcnow(), "is_default": True, "credit_cost": 700, "access_level": 1},
                ]
                
                self.db[self.COLLECTIONS['models']].insert_many(default_models)
                logger.info("Default models with credit system initialized in MongoDB")
        except Exception as e:
            logger.exception(f"Error initializing default models: {e}")
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    # =====================================
    # SUPPORTED MODELS METHODS (NEW)
    # =====================================
    
    def get_supported_models(self) -> Set[str]:
        """Get set of supported model names"""
        try:
            results = self.db[self.COLLECTIONS['models']].find({}, {"model_name": 1})
            return {doc["model_name"] for doc in results}
        except Exception as e:
            logger.exception(f"Error getting supported models: {e}")
            # Return default models as fallback
            return {"o3-mini", "gpt-4.1", "gpt-5", "gpt-oss-20b", "gpt-oss-120b"}
    
    def add_supported_model(self, model_name: str, credit_cost: int = 1, access_level: int = 0) -> tuple[bool, str]:
        """Add a new supported model with credit cost and access level"""
        try:
            model_name = model_name.strip()
            if not model_name:
                return False, "Model name cannot be empty"
                
            if credit_cost < 0:
                return False, "Credit cost cannot be negative"
                
            if access_level not in [0, 1, 2]:
                return False, "Access level must be 0, 1, or 2"
                
            # Check if model exists
            existing = self.db[self.COLLECTIONS['models']].find_one({"model_name": model_name})
            if existing:
                return False, f"Model '{model_name}' already exists"
                
            # Add new model with attributes
            result = self.db[self.COLLECTIONS['models']].insert_one({
                "model_name": model_name,
                "created_at": datetime.utcnow(),
                "is_default": False,
                "credit_cost": credit_cost,
                "access_level": access_level
            })
            
            if result.inserted_id:
                return True, f"Successfully added model '{model_name}' (Cost: {credit_cost}, Level: {access_level})"
            return False, "Failed to add model to database"
        except Exception as e:
            logger.exception(f"Error adding model {model_name}: {e}")
            return False, f"Database error: {e}"
    
    def remove_supported_model(self, model_name: str) -> tuple[bool, str]:
        """
        Remove a supported model
        Returns: (success: bool, message: str)
        """
        try:
            model_name = model_name.strip()
            if not model_name:
                return False, "Model name cannot be empty"
            
            # Check if model exists
            existing = self.db[self.COLLECTIONS['models']].find_one({"model_name": model_name})
            if not existing:
                return False, f"Model '{model_name}' does not exist"
            
            # Prevent removal of default models (optional safety check)
            if existing.get("is_default", False):
                return False, f"Cannot remove default model '{model_name}'"
            
            # Check if any users are currently using this model
            users_using_model = self.db[self.COLLECTIONS['user_config']].count_documents({"model": model_name})
            if users_using_model > 0:
                return False, f"Cannot remove model '{model_name}' - {users_using_model} user(s) are currently using it"
            
            # Remove the model
            result = self.db[self.COLLECTIONS['models']].delete_one({"model_name": model_name})
            
            if result.deleted_count > 0:
                return True, f"Successfully removed model '{model_name}'"
            else:
                return False, "Failed to remove model from database"
                
        except Exception as e:
            logger.exception(f"Error removing model {model_name}: {e}")
            return False, f"Database error: {e}"
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists in supported models"""
        try:
            result = self.db[self.COLLECTIONS['models']].find_one({"model_name": model_name})
            return result is not None
        except Exception as e:
            logger.exception(f"Error checking if model {model_name} exists: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model"""
        try:
            return self.db[self.COLLECTIONS['models']].find_one({"model_name": model_name})
        except Exception as e:
            logger.exception(f"Error getting model info for {model_name}: {e}")
            return None
    
    def list_all_models(self) -> List[Dict[str, Any]]:
        """Get list of all models with their details"""
        try:
            results = self.db[self.COLLECTIONS['models']].find({}).sort("created_at", 1)
            return list(results)
        except Exception as e:
            logger.exception(f"Error listing all models: {e}")
            return []
    
    def edit_supported_model(self, model_name: str, credit_cost: int = None, access_level: int = None) -> tuple[bool, str]:
        """Edit an existing model's settings"""
        try:
            model_name = model_name.strip()
            if not model_name:
                return False, "Model name cannot be empty"
                
            # Check if model exists
            existing = self.db[self.COLLECTIONS['models']].find_one({"model_name": model_name})
            if not existing:
                return False, f"Model '{model_name}' does not exist"
                
            # Prepare update data
            update_data = {"updated_at": datetime.utcnow()}
            if credit_cost is not None:
                if credit_cost < 0:
                    return False, "Credit cost cannot be negative"
                update_data["credit_cost"] = credit_cost
                
            if access_level is not None:
                if access_level not in [0, 1, 2]:
                    return False, "Access level must be 0, 1, or 2"
                update_data["access_level"] = access_level
                
            # Update the model
            result = self.db[self.COLLECTIONS['models']].update_one(
                {"model_name": model_name},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                return True, f"Successfully updated model '{model_name}'"
            return False, "No changes were made"
            
        except Exception as e:
            logger.exception(f"Error editing model {model_name}: {e}")
            return False, f"Database error: {e}"

    # =====================================
    # USER CONFIG METHODS (Updated to use database models)
    # =====================================
    
    def get_user_config(self, user_id: int) -> Dict[str, Any]:
        """Get user configuration"""
        try:
            result = self.db[self.COLLECTIONS['user_config']].find_one({"user_id": user_id})
            if result:
                # Validate that the user's model still exists
                user_model = result.get("model", "gemini-2.5-flash")
                if not self.model_exists(user_model):
                    # Model no longer exists, fallback to first available model
                    supported_models = self.get_supported_models()
                    if supported_models:
                        user_model = next(iter(supported_models))
                        self.set_user_config(user_id, model=user_model)
                    else:
                        user_model = "gemini-2.5-flash"  # Ultimate fallback

            return {
                "model": user_model,
                "system_prompt": result.get("system_prompt", "Tên của bạn là Ryuuko (nữ), nói tiếng việt"),
                "credit": result.get("credit", 0),
                "access_level": result.get("access_level", 0)
            }
        except Exception as e:
            logger.exception(f"Error getting user config for {user_id}: {e}")
            return {
                "model": "gemini-2.5-flash",
                "system_prompt": "Tên của bạn là Ryuuko (nữ), nói tiếng việt",
                "credit": 0,
                "access_level": 0
            }
    
    def set_user_config(self, user_id: int, model: Optional[str] = None, system_prompt: Optional[str] = None) -> bool:
        """Set user configuration"""
        try:
            update_data = {"updated_at": datetime.utcnow()}
            if model is not None:
                # Validate model exists
                if not self.model_exists(model):
                    logger.warning(f"Attempt to set non-existent model {model} for user {user_id}")
                    return False
                update_data["model"] = model
            if system_prompt is not None:
                update_data["system_prompt"] = system_prompt
            
            result = self.db[self.COLLECTIONS['user_config']].update_one(
                {"user_id": user_id},
                {
                    "$set": update_data,
                    "$setOnInsert": {
                        "user_id": user_id,
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            logger.exception(f"Error setting user config for {user_id}: {e}")
            return False
    
    def get_user_model(self, user_id: int) -> str:
        """Get user's preferred model"""
        config = self.get_user_config(user_id)
        return config["model"]
    
    def get_user_system_prompt(self, user_id: int) -> str:
        """Get user's system prompt"""
        config = self.get_user_config(user_id)
        return config["system_prompt"]
    
    def get_user_system_message(self, user_id: int) -> Dict[str, str]:
        """Get system message in OpenAI format"""
        return {
            "role": "system",
            "content": self.get_user_system_prompt(user_id)
        }
    
    # =====================================
    # MEMORY METHODS
    # =====================================
    
    def get_user_messages(self, user_id: int) -> List[Dict[str, str]]:
        """Get user's conversation history"""
        try:
            result = self.db[self.COLLECTIONS['memory']].find_one({"user_id": user_id})
            if result and "messages" in result:
                return result["messages"]
            return []
        except Exception as e:
            logger.exception(f"Error getting messages for user {user_id}: {e}")
            return []
    
    def add_message(self, user_id: int, message: Dict[str, str], max_messages: int = 50, max_tokens: int = 2000):
        """Add message to user's conversation history"""
        try:
            # Get current messages
            current_messages = self.get_user_messages(user_id)
            current_messages.append(message)
            
            # Prune messages if needed
            current_messages = self._prune_messages(current_messages, max_messages, max_tokens)
            
            # Update in database
            result = self.db[self.COLLECTIONS['memory']].update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "messages": current_messages,
                        "updated_at": datetime.utcnow()
                    },
                    "$setOnInsert": {
                        "user_id": user_id,
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            logger.exception(f"Error adding message for user {user_id}: {e}")
            return False
    
    def clear_user_memory(self, user_id: int) -> bool:
        """Clear user's conversation history"""
        try:
            result = self.db[self.COLLECTIONS['memory']].delete_one({"user_id": user_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.exception(f"Error clearing memory for user {user_id}: {e}")
            return False
    
    def _prune_messages(self, messages: List[Dict[str, str]], max_messages: int, max_tokens: int) -> List[Dict[str, str]]:
        """Prune messages based on count and token limits"""
        # Remove oldest messages if over limit
        while len(messages) > max_messages:
            messages.pop(0)
        
        # Remove oldest messages if over token limit
        total_tokens = sum(len(self.tokenizer.encode(msg["content"])) for msg in messages)
        while total_tokens > max_tokens and messages:
            removed = messages.pop(0)
            total_tokens -= len(self.tokenizer.encode(removed["content"]))
        
        return messages
    
    # =====================================
    # AUTHORIZED USERS METHODS
    # =====================================
    
    def get_authorized_users(self) -> Set[int]:
        """Get set of authorized user IDs"""
        try:
            results = self.db[self.COLLECTIONS['authorized']].find({}, {"user_id": 1})
            return {doc["user_id"] for doc in results}
        except Exception as e:
            logger.exception(f"Error getting authorized users: {e}")
            return set()
    
    def add_authorized_user(self, user_id: int) -> bool:
        """Add user to authorized list"""
        try:
            result = self.db[self.COLLECTIONS['authorized']].update_one(
                {"user_id": user_id},
                {
                    "$setOnInsert": {
                        "user_id": user_id,
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            logger.exception(f"Error adding authorized user {user_id}: {e}")
            return False
    
    def remove_authorized_user(self, user_id: int) -> bool:
        """Remove user from authorized list"""
        try:
            result = self.db[self.COLLECTIONS['authorized']].delete_one({"user_id": user_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.exception(f"Error removing authorized user {user_id}: {e}")
            return False
    
    def is_user_authorized(self, user_id: int) -> bool:
        """Check if user is authorized"""
        try:
            result = self.db[self.COLLECTIONS['authorized']].find_one({"user_id": user_id})
            return result is not None
        except Exception as e:
            logger.exception(f"Error checking authorization for user {user_id}: {e}")
            return False
    
    # =====================================
    # USER LEVEL AND CREDIT METHODS (NEW)
    # =====================================
    
    def set_user_level(self, user_id: int, level: int) -> bool:
        """Set user access level"""
        try:
            if level not in [0, 1, 2]:
                return False
                
            result = self.db[self.COLLECTIONS['user_config']].update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "access_level": level,
                        "updated_at": datetime.utcnow()
                    },
                    "$setOnInsert": {
                        "user_id": user_id,
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            logger.exception(f"Error setting level for user {user_id}: {e}")
            return False
    
    def add_user_credit(self, user_id: int, amount: int) -> tuple[bool, int]:
        """Add credit to user balance. Returns (success, new_balance)"""
        try:
            result = self.db[self.COLLECTIONS['user_config']].find_one_and_update(
                {"user_id": user_id},
                {
                    "$inc": {"credit": amount},
                    "$setOnInsert": {
                        "user_id": user_id,
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True,
                return_document=True
            )
            return True, result.get("credit", 0)
        except Exception as e:
            logger.exception(f"Error adding credit for user {user_id}: {e}")
            return False, 0
    
    def deduct_user_credit(self, user_id: int, amount: int) -> tuple[bool, int]:
        """Deduct credit from user balance. Returns (success, remaining_balance)"""
        try:
            result = self.db[self.COLLECTIONS['user_config']].find_one_and_update(
                {
                    "user_id": user_id,
                    "credit": {"$gte": amount}
                },
                {
                    "$inc": {"credit": -amount}
                },
                return_document=True
            )
            if result:
                return True, result.get("credit", 0)
            return False, 0
        except Exception as e:
            logger.exception(f"Error deducting credit for user {user_id}: {e}")
            return False, 0

# Singleton instance
_mongodb_store: Optional[MongoDBStore] = None

def get_mongodb_store() -> MongoDBStore:
    """Get singleton MongoDB store instance"""
    global _mongodb_store
    if _mongodb_store is None:
        raise RuntimeError("MongoDB store not initialized. Call init_mongodb_store() first.")
    return _mongodb_store

def init_mongodb_store(connection_string: str, database_name: str = "discord_openai_proxy") -> MongoDBStore:
    """Initialize MongoDB store singleton"""
    global _mongodb_store
    if _mongodb_store is None:
        _mongodb_store = MongoDBStore(connection_string, database_name)
    return _mongodb_store

def close_mongodb_store():
    """Close MongoDB store connection"""
    global _mongodb_store
    if _mongodb_store:
        _mongodb_store.close()
        _mongodb_store = None