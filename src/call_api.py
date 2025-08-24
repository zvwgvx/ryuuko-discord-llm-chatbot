import logging
import os
from typing import List, Dict, Any, Tuple, Optional
from openai import OpenAI
import google.generativeai as genai
import load_config

logger = logging.getLogger("discord-openai-proxy.call_api")

class APIClients:
    """Singleton class to manage API clients"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Initialize OpenAI client
        self.openai_client = self._init_openai_client()
        
        # Initialize Gemini clients
        self.gemini_available = False
        self.owner_gemini_available = False
        self.gemini_error = None
        self._init_gemini_clients()
        
        self._initialized = True
    
    def _init_openai_client(self) -> OpenAI:
        """Initialize OpenAI client with proxy support"""
        try:
            return OpenAI(
                api_key=load_config.OPENAI_API_KEY, 
                base_url=load_config.OPENAI_API_BASE
            )
        except TypeError:
            return OpenAI(api_key=load_config.OPENAI_API_KEY)
    
    def _init_gemini_clients(self) -> None:
        """Initialize Gemini API clients"""
        try:
            # Check regular client
            if hasattr(load_config, 'CLIENT_GEMINI_API_KEY') and load_config.CLIENT_GEMINI_API_KEY:
                self.gemini_available = True
                logger.info("Regular Gemini client available")
            else:
                self.gemini_error = "Client Gemini API key not found in config"
                logger.warning(self.gemini_error)

            # Check owner client
            if hasattr(load_config, 'OWNER_GEMINI_API_KEY') and load_config.OWNER_GEMINI_API_KEY:
                self.owner_gemini_available = True
                logger.info("Owner Gemini API key available")
            else:
                logger.warning("Owner Gemini API key not found, will use regular key for owner requests")

        except ImportError:
            self.gemini_error = "Google GenAI library not installed. Please run: pip install google-generativeai"
            logger.error(self.gemini_error)
        except Exception as e:
            self.gemini_error = f"Error initializing Gemini clients: {e}"
            logger.error(self.gemini_error)

# Global instance
clients = APIClients()

class GeminiConfig:
    """Configuration class for Gemini API parameters"""
    
    DEFAULT_SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]
    
    DEFAULT_GENERATION_CONFIG = {
        "temperature": 1.2,
        "top_p": 1.0,
        "top_k": 40,
    }
    
    FINISH_REASONS = {
        1: "STOP",
        2: "MAX_TOKENS", 
        3: "SAFETY",
        4: "RECITATION",
        5: "OTHER"
    }

def is_gemini_model(model_name: str) -> bool:
    """Check if the model is a Gemini model"""
    return model_name.startswith("gemini-") or model_name.startswith("gemma-")

def build_gemini_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Build a comprehensive prompt from message history
    FIXED VERSION: Better context separation and current message emphasis
    """
    prompt_parts = []
    current_message = None
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if not content.strip():
            continue
            
        if role == "system":
            prompt_parts.append(f"System Instructions: {content}")
        elif role == "user":
            # Check if this is the last message (current request)
            if i == len(messages) - 1:
                current_message = content
            else:
                prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    
    if not prompt_parts and not current_message:
        return "Hello"
    
    # Build the final prompt with clear separation
    if len(prompt_parts) > 0 and current_message:
        # We have conversation history + current message
        context = "\n\n".join(prompt_parts)
        return f"{context}\n\n---\n\nUser (CURRENT REQUEST): {current_message}\n\nPlease respond to this current request:"
    elif current_message:
        # Only current message, no history
        return f"User: {current_message}"
    elif len(prompt_parts) > 0:
        # Only history, no current message (shouldn't happen)
        return "\n\n".join(prompt_parts)
    
    return "Hello"

def create_generation_config(
    temperature: Optional[float] = None,
    top_p: Optional[float] = None, 
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """Create generation config with custom parameters"""
    config = GeminiConfig.DEFAULT_GENERATION_CONFIG.copy()
    
    # Update with provided parameters
    if temperature is not None:
        config["temperature"] = max(0.0, min(2.0, temperature))
    if top_p is not None:
        config["top_p"] = max(0.0, min(1.0, top_p))
    if top_k is not None:
        config["top_k"] = max(1, min(100, top_k))
    if max_output_tokens is not None:
        config["max_output_tokens"] = max(1, min(32768, max_output_tokens))
    
    return config

def extract_gemini_response(response) -> Tuple[bool, str]:
    """Extract text from Gemini response with comprehensive error handling"""
    try:
        # Check if response has candidates
        if not hasattr(response, 'candidates') or not response.candidates:
            return False, "No candidates returned from Gemini API"
        
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, 'finish_reason', None)
        
        # Handle finish reasons
        if finish_reason is not None:
            reason_name = GeminiConfig.FINISH_REASONS.get(finish_reason, f"UNKNOWN({finish_reason})")
            logger.debug(f"Gemini finish reason: {reason_name}")
            
            if finish_reason == 3:  # SAFETY
                return False, "Response blocked by content safety filters. Please rephrase your request."
            elif finish_reason == 4:  # RECITATION
                return False, "Response blocked due to recitation concerns. Please rephrase your request."
            elif finish_reason not in [1, 2]:  # Not STOP or MAX_TOKENS
                return False, f"Response generation stopped unexpectedly (reason: {reason_name})"
        
        # Extract text content
        text_content = ""
        if (hasattr(candidate, 'content') and candidate.content and 
            hasattr(candidate.content, 'parts') and candidate.content.parts):
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    text_content += part.text
        
        # Return result
        if text_content and text_content.strip():
            return True, text_content.strip()
        
        # Handle empty responses
        if finish_reason == 1:  # STOP
            return False, "Gemini returned an empty response"
        
        return False, f"No valid content returned (reason: {GeminiConfig.FINISH_REASONS.get(finish_reason, 'Unknown')})"
        
    except Exception as e:
        logger.exception(f"Error extracting response text: {e}")
        return False, f"Error processing Gemini response: {str(e)}"

def call_gemini_api(
    messages: List[Dict[str, str]], 
    model: str, 
    is_owner: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Call Gemini API with enhanced parameter support
    
    Args:
        messages: List of message dictionaries
        model: Model name (e.g., 'gemini-pro')
        is_owner: Whether to use owner API key
        temperature: Controls randomness (0.0-2.0)
        top_p: Controls nucleus sampling (0.0-1.0) 
        top_k: Controls top-k sampling (1-100)
        max_output_tokens: Maximum tokens in response (1-32768)
    """
    try:
        # Configure API key
        if is_owner and clients.owner_gemini_available and hasattr(load_config, 'OWNER_GEMINI_API_KEY'):
            genai.configure(api_key=load_config.OWNER_GEMINI_API_KEY)
            logger.debug("Using owner's Gemini API key")
        elif clients.gemini_available:
            genai.configure(api_key=load_config.CLIENT_GEMINI_API_KEY)
            logger.debug("Using regular Gemini API key")
        else:
            return False, clients.gemini_error or "Gemini API not initialized"

        # Create generation config
        generation_config = create_generation_config(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens
        )
        
        logger.debug(f"Gemini generation config: {generation_config}")
        
        # Create model instance
        model_instance = genai.GenerativeModel(
            model,
            safety_settings=GeminiConfig.DEFAULT_SAFETY_SETTINGS,
            generation_config=generation_config
        )

        # Build prompt and generate response
        prompt = build_gemini_prompt(messages)
        logger.debug(f"Gemini prompt length: {len(prompt)} characters")
        
        response = model_instance.generate_content(prompt)
        
        return extract_gemini_response(response)

    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["harm", "safety", "blocked"]):
            return False, "Response blocked by content safety filters. Please rephrase your request."
        
        logger.exception(f"Error calling Gemini API: {e}")
        return False, f"Gemini API error: {str(e)}"

def is_model_available(model: str) -> Tuple[bool, str]:
    """Check if a model is available for use"""
    if is_gemini_model(model):
        if not clients.gemini_available:
            return False, clients.gemini_error or "Gemini API is not available"
        return True, ""
    return True, ""  # Non-Gemini models are assumed available

def call_openai_proxy(
    messages: List[Dict[str, str]], 
    model: str = "gpt-3.5-turbo", 
    is_owner: bool = False,
    # Gemini-specific parameters (ignored for OpenAI models)
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Call OpenAI API or Gemini API based on model type
    
    Args:
        messages: List of message dictionaries
        model: Model name
        is_owner: Whether to use owner privileges
        temperature: Temperature for generation (Gemini only)
        top_p: Top-p sampling parameter (Gemini only)
        top_k: Top-k sampling parameter (Gemini only)  
        max_output_tokens: Maximum output tokens (Gemini only)
    
    Returns:
        Tuple of (success: bool, response: str)
    """
    try:
        # Check model availability
        available, error = is_model_available(model)
        if not available:
            return False, error

        if is_gemini_model(model):
            return call_gemini_api(
                messages=messages,
                model=model,
                is_owner=is_owner,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        else:
            # OpenAI API call (parameters like top_k ignored)
            openai_temperature = temperature if temperature is not None else 1.1
            
            response = clients.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=openai_temperature,
                timeout=load_config.REQUEST_TIMEOUT
            )

            choice = response.choices[0]
            if choice.finish_reason == "length":
                logger.warning(f"Response truncated due to max_tokens limit for model {model}")

            return True, choice.message.content

    except Exception as e:
        logger.exception(f"Error calling API for model {model}: {e}")
        return False, str(e)