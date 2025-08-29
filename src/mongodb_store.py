#!/usr/bin/env python3
# coding: utf-8
# mongodb_store.py - FIXED VERSION - Resolves memory disappearing issues

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from pymongo import MongoClient, WriteConcern
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import tiktoken
import time

logger = logging.getLogger("discord-openai-proxy.mongodb_store")

class MongoDBStore:
    """MongoDB storage manager for Discord OpenAI proxy"""
    
    def __init__(self, connection_string: str, database_name: str = "discord_openai_proxy"):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client: Optional[MongoClient] = None
        self.db = None
        
        # Use a more reliable tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4")
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Collection names
        self.COLLECTIONS = {
            'user_config': 'user_configs',
            'memory': 'user_memory', 
            'authorized': 'authorized_users',
            'models': 'supported_models',
            'pmodels': 'profile_models'  # New collection for profile models
        }
        
        self._connect()
        self._initialize_default_models()
    
    def _connect(self):
        """Establish MongoDB connection with proper write concern"""
        try:
            # Use write concern for data consistency
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
                w="majority",  # Wait for majority write acknowledgment
                journal=True         # Wait for journal acknowledgment
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
            
            # Memory indexes - add compound index for better performance
            self.db[self.COLLECTIONS['memory']].create_index("user_id", unique=True)
            self.db[self.COLLECTIONS['memory']].create_index([("user_id", 1), ("updated_at", -1)])
            
            # Authorized users indexes
            self.db[self.COLLECTIONS['authorized']].create_index("user_id", unique=True)
            
            # Supported models indexes
            self.db[self.COLLECTIONS['models']].create_index("model_name", unique=True)
            
            # Profile models indexes
            self.db[self.COLLECTIONS['pmodels']].create_index("name", unique=True)
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.exception(f"Error creating indexes: {e}")
    
    def _initialize_default_models(self):
        """Initialize default supported models if collection is empty"""
        try:
            count = self.db[self.COLLECTIONS['models']].count_documents({})
            if count == 0:
                default_models = [
                    {"model_name": "gemini-2.5-flash", "created_at": datetime.utcnow(), "is_default": True, "credit_cost": 10, "access_level": 0},
                    {"model_name": "gemini-2.5-pro", "created_at": datetime.utcnow(), "is_default": True, "credit_cost": 50, "access_level": 0},
                    {"model_name": "gpt-3.5-turbo", "created_at": datetime.utcnow(), "is_default": True, "credit_cost": 200, "access_level": 0},
                    {"model_name": "gpt-5", "created_at": datetime.utcnow(), "is_default": True, "credit_cost": 700, "access_level": 1}
                ]
                
                self.db[self.COLLECTIONS['models']].insert_many(default_models)
                logger.info("Default models with credit system initialized in MongoDB")
        except Exception as e:
            logger.exception(f"Error initializing default models: {e}")
    
    # =====================================
    # MEMORY METHODS - COMPLETELY FIXED
    # =====================================
    
    def get_user_messages(self, user_id: int) -> List[Dict[str, str]]:
        """Get user's conversation history with strong consistency"""
        try:
            # Force read from primary with session for consistency
            with self.client.start_session() as session:
                with session.start_transaction():
                    result = self.db[self.COLLECTIONS['memory']].find_one(
                        {"user_id": user_id},
                        session=session
                    )
            
            if result and "messages" in result:
                messages = result["messages"]
                logger.debug(f"Retrieved {len(messages)} messages for user {user_id}")
                return messages
            
            logger.debug(f"No messages found for user {user_id}")
            return []
        except Exception as e:
            logger.exception(f"Error getting messages for user {user_id}: {e}")
            return []

    def add_message(self, user_id: int, message: Dict[str, str], max_messages: int = 25, max_tokens: int = 4000):
        """
        Add message to user's conversation history - FIXED VERSION
        Uses atomic operations to prevent race conditions
        """
        try:
            current_time = datetime.utcnow()
            
            # Use transaction for consistency
            with self.client.start_session() as session:
                with session.start_transaction():
                    # First, get current messages to check limits
                    current_doc = self.db[self.COLLECTIONS['memory']].find_one(
                        {"user_id": user_id}, 
                        session=session
                    )
                    
                    current_messages = current_doc.get("messages", []) if current_doc else []
                    
                    # Add new message
                    updated_messages = current_messages + [message]
                    
                    # Apply limits
                    updated_messages = self._apply_memory_limits(updated_messages, max_messages, max_tokens)
                    
                    # Atomic upsert with the processed messages
                    result = self.db[self.COLLECTIONS['memory']].find_one_and_update(
                        {"user_id": user_id},
                        {
                            "$set": {
                                "messages": updated_messages,
                                "updated_at": current_time,
                                "message_count": len(updated_messages)
                            },
                            "$setOnInsert": {
                                "user_id": user_id,
                                "created_at": current_time
                            }
                        },
                        upsert=True,
                        return_document=True,
                        session=session
                    )
            
            logger.info(f"Successfully added message for user {user_id}. Total messages: {len(updated_messages)}")
            return True
            
        except Exception as e:
            logger.exception(f"Error adding message for user {user_id}: {e}")
            return False

    def _apply_memory_limits(self, messages: List[Dict[str, str]], max_messages: int, max_tokens: int) -> List[Dict[str, str]]:
        """Apply message count and token limits while preserving conversation flow"""
        if not messages:
            return messages
        
        # First apply message count limit
        if len(messages) > max_messages:
            # Keep the most recent messages
            messages = messages[-max_messages:]
        
        # Then apply token limit
        total_tokens = 0
        token_counts = []
        
        # Calculate tokens for each message
        for msg in messages:
            try:
                tokens = len(self.tokenizer.encode(msg.get("content", "")))
                token_counts.append(tokens)
                total_tokens += tokens
            except Exception as e:
                logger.warning(f"Error calculating tokens for message: {e}")
                token_counts.append(100)  # Fallback estimate
                total_tokens += 100
        
        # If over token limit, remove oldest messages
        if total_tokens > max_tokens:
            final_messages = []
            current_tokens = 0
            
            # Work backwards to keep most recent messages within token limit
            for i in range(len(messages) - 1, -1, -1):
                msg_tokens = token_counts[i]
                if current_tokens + msg_tokens <= max_tokens:
                    final_messages.insert(0, messages[i])
                    current_tokens += msg_tokens
                else:
                    break
            
            logger.debug(f"Trimmed messages from {len(messages)} to {len(final_messages)} due to token limit")
            return final_messages
        
        return messages

    def clear_user_memory(self, user_id: int) -> bool:
        """Clear user's conversation history with transaction"""
        try:
            with self.client.start_session() as session:
                with session.start_transaction():
                    result = self.db[self.COLLECTIONS['memory']].delete_one(
                        {"user_id": user_id},
                        session=session
                    )
            
            logger.info(f"Cleared memory for user {user_id}")
            return result.deleted_count > 0
        except Exception as e:
            logger.exception(f"Error clearing memory for user {user_id}: {e}")
            return False

    def remove_last_message(self, user_id: int) -> bool:
        """Remove the last message from user's conversation history"""
        try:
            with self.client.start_session() as session:
                with session.start_transaction():
                    # Get current messages
                    current_doc = self.db[self.COLLECTIONS['memory']].find_one(
                        {"user_id": user_id}, 
                        session=session
                    )
                    
                    if not current_doc or "messages" not in current_doc:
                        return False
                    
                    messages = current_doc["messages"]
                    if not messages:
                        return False
                    
                    # Remove last message
                    messages = messages[:-1]
                    
                    # Update document
                    result = self.db[self.COLLECTIONS['memory']].update_one(
                        {"user_id": user_id},
                        {
                            "$set": {
                                "messages": messages,
                                "updated_at": datetime.utcnow(),
                                "message_count": len(messages)
                            }
                        },
                        session=session
                    )
            
            return result.modified_count > 0
        except Exception as e:
            logger.exception(f"Error removing last message for user {user_id}: {e}")
            return False

    # =====================================
    # USER CONFIG METHODS
    # =====================================
    
    def get_user_config(self, user_id: int) -> Dict[str, Any]:
        """Get user configuration"""
        try:
            result = self.db[self.COLLECTIONS['user_config']].find_one({"user_id": user_id})
            if result:
                user_model = result.get("model", "gemini-2.5-flash")
                if not self.model_exists(user_model):
                    supported_models = self.get_supported_models()
                    if supported_models:
                        user_model = next(iter(supported_models))
                        self.set_user_config(user_id, model=user_model)
                    else:
                        user_model = "gemini-2.5-flash"
                
                return {
                    "model": user_model,
                    "system_prompt": result.get("system_prompt", "Tên của bạn là Ryuuko (nữ), nói tiếng việt"),
                    "credit": result.get("credit", 0),
                    "access_level": result.get("access_level", 0)
                }
            else:
                return {
                    "model": "gemini-2.5-flash",
                    "system_prompt": "Tên của bạn là Ryuuko (nữ), nói tiếng việt",
                    "credit": 0,
                    "access_level": 0
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
    # SUPPORTED MODELS METHODS
    # =====================================
    
    def get_supported_models(self) -> Set[str]:
        """Get set of supported model names"""
        try:
            results = self.db[self.COLLECTIONS['models']].find({}, {"model_name": 1})
            return {doc["model_name"] for doc in results}
        except Exception as e:
            logger.exception(f"Error getting supported models: {e}")
            return {"gemini-2.5-flash", "gemini-2.5-pro", "gpt-3.5-turbo", "gpt-5"}
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists in supported models or profile models"""
        try:
            # Check regular models
            model = self.db[self.COLLECTIONS['models']].find_one({"model_name": model_name})
            if model:
                return True
            
            # Check profile models
            pmodel = self.db[self.COLLECTIONS['pmodels']].find_one({"name": model_name})
            return pmodel is not None
        
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
    
    def add_supported_model(self, model_name: str, credit_cost: int = 1, access_level: int = 0) -> tuple[bool, str]:
        """Add a new supported model"""
        try:
            model_name = model_name.strip()
            if not model_name:
                return False, "Model name cannot be empty"
                
            if credit_cost < 0:
                return False, "Credit cost cannot be negative"
                
            if access_level not in [0, 1, 2, 3]:
                return False, "Access level must be 0, 1, 2, or 3"
                
            existing = self.db[self.COLLECTIONS['models']].find_one({"model_name": model_name})
            if existing:
                return False, f"Model '{model_name}' already exists"
                
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
        """Remove a supported model"""
        try:
            model_name = model_name.strip()
            if not model_name:
                return False, "Model name cannot be empty"
            
            existing = self.db[self.COLLECTIONS['models']].find_one({"model_name": model_name})
            if not existing:
                return False, f"Model '{model_name}' does not exist"
            
            if existing.get("is_default", False):
                return False, f"Cannot remove default model '{model_name}'"
            
            users_using_model = self.db[self.COLLECTIONS['user_config']].count_documents({"model": model_name})
            if users_using_model > 0:
                return False, f"Cannot remove model '{model_name}' - {users_using_model} user(s) are currently using it"
            
            result = self.db[self.COLLECTIONS['models']].delete_one({"model_name": model_name})
            
            if result.deleted_count > 0:
                return True, f"Successfully removed model '{model_name}'"
            else:
                return False, "Failed to remove model from database"
                
        except Exception as e:
            logger.exception(f"Error removing model {model_name}: {e}")
            return False, f"Database error: {e}"
    
    def edit_supported_model(self, model_name: str, credit_cost: int = None, access_level: int = None) -> tuple[bool, str]:
        """Edit an existing model's settings"""
        try:
            model_name = model_name.strip()
            if not model_name:
                return False, "Model name cannot be empty"
                
            existing = self.db[self.COLLECTIONS['models']].find_one({"model_name": model_name})
            if not existing:
                return False, f"Model '{model_name}' does not exist"
                
            update_data = {"updated_at": datetime.utcnow()}
            if credit_cost is not None:
                if credit_cost < 0:
                    return False, "Credit cost cannot be negative"
                update_data["credit_cost"] = credit_cost
                
            if access_level is not None:
                if access_level not in [0, 1, 2, 3]:
                    return False, "Access level must be 0, 1, 2, or 3"
                update_data["access_level"] = access_level
                
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
    # USER LEVEL AND CREDIT METHODS
    # =====================================
    
    def set_user_level(self, user_id: int, level: int) -> bool:
        """Set user access level"""
        try:
            if level not in [0, 1, 2, 3]:
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
        """Add credit to user balance"""
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
        """Deduct credit from user balance"""
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

    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

    # =====================================
    # PROFILE MODEL METHODS
    # =====================================
    
    def add_profile_model(self, name: str) -> tuple[bool, str]:
        """Add a new profile model with default values"""
        try:
            name = name.strip()
            if not name:
                return False, "Profile model name cannot be empty"
            
            # Check if profile already exists
            existing = self.db[self.COLLECTIONS['pmodels']].find_one({"name": name})
            if existing:
                return False, f"Profile model '{name}' already exists"
            
            # Create profile with default values
            profile = {
                "name": name,
                "base_model": None,
                "sys_prompt": None,
                "credit_cost": 0,
                "access_level": 0,
                "is_live": False,
                "created_at": datetime.utcnow()
            }
            
            result = self.db[self.COLLECTIONS['pmodels']].insert_one(profile)
            
            if result.inserted_id:
                return True, f"Successfully created profile model '{name}'. Use ;edit pmodel to configure settings."
            return False, "Failed to create profile model"
            
        except Exception as e:
            logger.exception(f"Error adding profile model {name}: {e}")
            return False, f"Database error: {e}"

    def get_profile_model_details(self, name: str) -> tuple[bool, str]:
        """Get detailed info about a profile model"""
        try:
            profile = self.db[self.COLLECTIONS['pmodels']].find_one({"name": name})
            if not profile:
                return False, f"Profile model '{name}' does not exist"

            # Format details
            details = [
                f"**Profile Model: `{name}`**",
                f"Base Model: `{profile.get('base_model') or 'Not set'}`",
                f"Cost: {profile.get('credit_cost', 0)} credits",
                f"Level: {profile.get('access_level', 0)}",
                f"Live: {'✅' if profile.get('is_live') else '❌'}"
            ]
            
            # Add system prompt at the end
            sys_prompt = profile.get('sys_prompt')
            if sys_prompt:
                details.append("\n**System Prompt:**")
                details.append(f"`{sys_prompt}`")
            
            return True, "\n".join(details)
        except Exception as e:
            logger.exception(f"Error getting profile model details {name}: {e}")
            return False, f"Error: {e}"
    
    def edit_profile_model(self, name: str, field: str, value: Any) -> tuple[bool, str]:
        """Edit a profile model field"""
        try:
            name = name.strip()
            if not name:
                return False, "Profile model name cannot be empty"
            
            # Get existing profile
            profile = self.db[self.COLLECTIONS['pmodels']].find_one({"name": name})
            if not profile:
                return False, f"Profile model '{name}' does not exist"
            
            # Validate field
            valid_fields = ["base_model", "sys_prompt", "credit_cost", "access_level", "is_live"]
            if field not in valid_fields:
                return False, f"Invalid field '{field}'. Valid fields: {', '.join(valid_fields)}"
            
            # Validate and convert value based on field
            if field == "base_model":
                # Remove model existence check - allow any model name
                pass
            elif field in ["credit_cost", "access_level"]:
                try:
                    value = int(value)
                    if field == "access_level" and value not in [0, 1, 2, 3]:
                        return False, "Access level must be 0, 1, 2, or 3"
                    if field == "credit_cost" and value < 0:
                        return False, "Credit cost cannot be negative"
                except ValueError:
                    return False, f"{field} must be an integer"
            elif field == "is_live":
                value = str(value).lower() in ['true', '1', 'yes']
            
            # Update field
            result = self.db[self.COLLECTIONS['pmodels']].update_one(
                {"name": name},
                {
                    "$set": {
                        field: value,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            if result.modified_count > 0:
                return True, f"Successfully updated {field} for profile model '{name}'"
            return False, "No changes were made"
            
        except Exception as e:
            logger.exception(f"Error editing profile model {name}: {e}")
            return False, f"Database error: {e}"
            
    def get_profile_model(self, name: str) -> Optional[Dict[str, Any]]:
        """Get profile model details"""
        try:
            return self.db[self.COLLECTIONS['pmodels']].find_one({"name": name})
        except Exception as e:
            logger.exception(f"Error getting profile model {name}: {e}")
            return None
            
    def list_profile_models(self) -> List[Dict[str, Any]]:
        """List all profile models"""
        try:
            return list(self.db[self.COLLECTIONS['pmodels']].find().sort("name", 1))
        except Exception as e:
            logger.exception("Error listing profile models")
            return []
    
    def delete_profile_model(self, name: str) -> bool:
        """Delete a profile model"""
        try:
            result = self.db[self.COLLECTIONS['pmodels']].delete_one({"name": name})
            return result.deleted_count > 0
        except Exception as e:
            logger.exception(f"Error deleting profile model {name}: {e}")
            return False
        
    def get_users_using_model(self, model_name: str) -> List[int]:
        """Get list of user IDs currently using a specific model"""
        try:
            results = self.db[self.COLLECTIONS['user_config']].find(
                {"model": model_name},
                {"user_id": 1}
            )
            return [doc["user_id"] for doc in results]
        except Exception as e:
            logger.exception(f"Error getting users for model {model_name}: {e}")
            return []

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