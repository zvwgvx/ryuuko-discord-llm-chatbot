import asyncio
from typing import List, Dict, Any, Tuple, Optional, AsyncIterator, Union
from openai import OpenAI
import google.generativeai as genai
import load_config
from mongodb_store import get_mongodb_store
from ratelimit import get_rate_limiter
import logging

# Add Live API imports
try:
    from google import genai as live_genai
    from google.genai import types
    LIVE_API_AVAILABLE = True
except ImportError:
    LIVE_API_AVAILABLE = False

logger = logging.getLogger("call_api")

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
        
        # Initialize Live API client
        self.live_client = None
        self.live_available = False
        self.live_error = None
        self._init_live_client()
        
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
    
    def _init_live_client(self) -> None:
        """Initialize Gemini Live API client"""
        try:
            if not LIVE_API_AVAILABLE:
                self.live_error = "Google GenAI SDK not installed. Run: pip install google-genai"
                logger.error(self.live_error)
                return
            
            # Use same API key as regular Gemini
            api_key = None
            if hasattr(load_config, 'CLIENT_GEMINI_API_KEY') and load_config.CLIENT_GEMINI_API_KEY:
                api_key = load_config.CLIENT_GEMINI_API_KEY
            elif hasattr(load_config, 'OWNER_GEMINI_API_KEY') and load_config.OWNER_GEMINI_API_KEY:
                api_key = load_config.OWNER_GEMINI_API_KEY
            
            if api_key:
                self.live_client = live_genai.Client(
                    http_options={"api_version": "v1beta"},
                    api_key=api_key,
                )
                self.live_available = True
                logger.info("Gemini Live API client initialized")
            else:
                self.live_error = "No API key found for Live API"
                logger.warning(self.live_error)
                
        except Exception as e:
            self.live_error = f"Error initializing Live API client: {e}"
            logger.error(self.live_error)

# Global instances
clients = APIClients()
rate_limiter = get_rate_limiter()

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
    
    # New: Tools and thinking configuration for regular Gemini models
    DEFAULT_TOOLS_CONFIG = {
        "enable_tools": True,
        "enable_thinking": True,
        "thinking_budget": 16000,
        "media_resolution": "MEDIA_RESOLUTION_MEDIUM"
    }
    
    FINISH_REASONS = {
        1: "STOP",
        2: "MAX_TOKENS", 
        3: "SAFETY",
        4: "RECITATION",
        5: "OTHER"
    }

def get_gemini_tools():
    """Create and return Gemini tools configuration for regular models"""
    try:
        if not LIVE_API_AVAILABLE:
            logger.warning("Google GenAI SDK not available for tools, using basic tools")
            # Return basic tools that work with google-generativeai
            return []
            
        tools = [
            types.Tool(url_context=types.UrlContext()),
            types.Tool(code_execution=types.ToolCodeExecution()),
            types.Tool(google_search=types.GoogleSearch()),
        ]
        logger.debug("Gemini tools configured: URL Context, Code Execution, and Google Search")
        return tools
    except Exception as e:
        logger.error(f"Error creating Gemini tools: {e}")
        return []

def get_live_api_tools():
    """Create and return Live API tools configuration"""
    try:
        if not LIVE_API_AVAILABLE:
            logger.warning("Live API not available, returning empty tools list")
            return []
            
        tools = [
            types.Tool(url_context=types.UrlContext()),
            types.Tool(google_search=types.GoogleSearch()),
        ]
        logger.debug("Live API tools configured: URL Context and Google Search")
        return tools
    except Exception as e:
        logger.error(f"Error creating Live API tools: {e}")
        return []

def is_gemini_live_model(model_name: str) -> bool:
    """Check if the model requires Live API"""
    live_model_patterns = [
        "gemini-2.5-flash-live-preview",
        "gemini-2.5-flash-preview-native-audio-dialog",
        "gemini-2.5-flash-exp-native-audio-thinking-dialog",
        "live-preview"
    ]
    
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in live_model_patterns)

def is_gemini_model(model_name: str) -> bool:
    """Check if the model is a Gemini model"""
    return (model_name.startswith("gemini-") or 
            model_name.startswith("gemma-") or 
            "live-preview" in model_name)

def is_thinking_model(model_name: str) -> bool:
    """Check if the model supports thinking (reasoning)"""
    thinking_patterns = [
        "thinking",
        "reasoning", 
        "gemini-2.0-flash-thinking-exp",
        "gemini-2.5-flash-thinking"
    ]
    
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in thinking_patterns)

def process_multimodal_content(content: Union[str, List[Dict]]) -> Union[str, List]:
    """
    Process multimodal content for Gemini API
    
    Args:
        content: Either a string or list of content parts (text + images)
        
    Returns:
        Processed content for Gemini API
    """
    try:
        if isinstance(content, str):
            # Simple text content
            return content
            
        if isinstance(content, list):
            # Multimodal content - process each part
            parts = []
            
            for part in content:
                if not isinstance(part, dict):
                    continue
                    
                part_type = part.get("type", "")
                
                if part_type == "text":
                    # Text part
                    text_content = part.get("text", "")
                    if text_content:
                        parts.append(text_content)
                        
                elif part_type == "image":
                    # Image part - convert to Gemini format
                    image_data = part.get("image_data", {})
                    if image_data:
                        data = image_data.get("data", "")
                        mime_type = image_data.get("mime_type", "image/jpeg")
                        
                        if data:
                            # Create image part for Gemini
                            import base64
                            image_bytes = base64.b64decode(data)
                            
                            # Create PIL Image for Gemini
                            try:
                                from PIL import Image
                                import io
                                
                                image = Image.open(io.BytesIO(image_bytes))
                                parts.append(image)
                                logger.debug(f"Added image to Gemini content: {mime_type}")
                                
                            except ImportError:
                                logger.warning("PIL not available, using raw image data")
                                # Fallback: use dict format that Gemini accepts
                                parts.append({
                                    'mime_type': mime_type,
                                    'data': data
                                })
                            except Exception as e:
                                logger.error(f"Error processing image: {e}")
                                continue
            
            if parts:
                return parts
            else:
                return "Hello"  # Fallback if no valid parts
        
        # Fallback for unknown content type
        logger.warning(f"Unknown content type: {type(content)}")
        return str(content) if content else "Hello"
        
    except Exception as e:
        logger.error(f"Error processing multimodal content: {e}")
        return "Hello"  # Safe fallback

def build_gemini_prompt_multimodal(messages: List[Dict[str, Any]]) -> List:
    """
    Build a multimodal prompt for Gemini API that supports images
    
    Args:
        messages: List of message dictionaries with potentially multimodal content
        
    Returns:
        List of content parts for Gemini API
    """
    try:
        # Convert messages to Gemini format
        contents = []
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Skip empty messages
            if not content:
                continue
            
            # Process system messages differently
            if role == "system":
                # System message - add as text instruction
                if isinstance(content, str):
                    contents.append(f"System Instructions: {content}")
                continue
            
            # Process user/assistant messages
            if role in ["user", "assistant"]:
                processed_content = process_multimodal_content(content)
                
                # Add role prefix for context if it's text
                if isinstance(processed_content, str):
                    # Add role context for text-only messages
                    if role == "user":
                        if i == len(messages) - 1:
                            # Current request - emphasize
                            contents.append(f"User (CURRENT REQUEST): {processed_content}")
                        else:
                            contents.append(f"User: {processed_content}")
                    else:  # assistant
                        contents.append(f"Assistant: {processed_content}")
                else:
                    # Multimodal content - add directly
                    contents.extend(processed_content if isinstance(processed_content, list) else [processed_content])
        
        # If no contents, return basic prompt
        if not contents:
            return ["Hello"]
        
        return contents
        
    except Exception as e:
        logger.error(f"Error building multimodal prompt: {e}")
        return ["Hello"]

def build_gemini_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Build a comprehensive prompt from message history (text-only version)
    FIXED VERSION: Better context separation and current message emphasis
    """
    prompt_parts = []
    current_message = None
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        # Fix: Check if content is None or empty string
        if content is None or not content.strip():
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

def has_multimodal_content(messages: List[Dict[str, Any]]) -> bool:
    """Check if any message contains multimodal content (images)"""
    try:
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Check if any part is an image
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        return True
        return False
    except Exception as e:
        logger.error(f"Error checking multimodal content: {e}")
        return False

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

def create_gemini_generate_content_config(
    model: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    enable_tools: bool = True,
    enable_thinking: bool = True,
    thinking_budget: int = 16000
) -> Optional[Any]:
    """Create GenerateContentConfig for regular Gemini models with tools and thinking support"""
    try:
        if not LIVE_API_AVAILABLE:
            logger.debug("SDK not available, using traditional generation config")
            return None
            
        # Get tools if enabled
        tools = get_gemini_tools() if enable_tools else []
        
        # Build config parameters
        config_params = {
            "temperature": temperature if temperature is not None else 1.1,
            "media_resolution": GeminiConfig.DEFAULT_TOOLS_CONFIG["media_resolution"]
        }
        
        # Add tools if available
        if tools:
            config_params["tools"] = tools
        
        # Add thinking config for models that support it
        if enable_thinking and is_thinking_model(model):
            config_params["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
            logger.debug(f"Thinking config added for model {model} with budget {thinking_budget}")
        
        # Add other parameters if specified
        if top_p is not None:
            config_params["top_p"] = max(0.0, min(1.0, top_p))
        if top_k is not None:
            config_params["top_k"] = max(1, min(100, top_k))
        if max_output_tokens is not None:
            config_params["max_output_tokens"] = max(1, min(32768, max_output_tokens))
        
        config = types.GenerateContentConfig(**config_params)
        logger.debug(f"Created GenerateContentConfig with tools: {len(tools)}, thinking: {enable_thinking and is_thinking_model(model)}")
        
        return config
        
    except Exception as e:
        logger.error(f"Error creating GenerateContentConfig: {e}")
        return None

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

def call_gemini_live_api_sync(
    messages: List[Dict[str, str]], 
    model: str,
    is_owner: bool = False
) -> Tuple[bool, str]:
    """Synchronous wrapper for Live API"""
    try:
        return asyncio.run(call_gemini_live_api_async(messages, model, is_owner))
    except Exception as e:
        logger.exception(f"Error in Live API sync wrapper: {e}")
        return False, f"Live API error: {str(e)}"

async def call_gemini_live_api_async(
    messages: List[Dict[str, str]], 
    model: str,
    is_owner: bool = False
) -> Tuple[bool, str]:
    """Call Gemini Live API with tools support"""
    
    if not clients.live_available:
        return False, clients.live_error or "Gemini Live API not available"
    
    try:
        # Apply rate limiting
        await rate_limiter.wait_if_needed(model)
        
        # Get Live API tools
        tools = get_live_api_tools()
        
        # Create config with tools enabled
        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            media_resolution="MEDIA_RESOLUTION_LOW",
            tools=tools if tools else None,  # Only include tools if they exist
        )
        
        # Convert messages to simple prompt
        prompt = build_gemini_prompt(messages)
        logger.debug(f"Live API prompt length: {len(prompt)} characters")
        logger.debug(f"Live API tools enabled: {len(tools) if tools else 0}")
        
        # Map model names to correct Live API model names
        model_mapping = {
            "gemini-2.5-flash-live-preview": "models/gemini-2.5-flash-live-preview",
            "gemini-2.5-flash-preview-native-audio-dialog": "models/gemini-2.5-flash-preview-native-audio-dialog",
            "gemini-2.5-flash-exp-native-audio-thinking-dialog": "models/gemini-2.5-flash-exp-native-audio-thinking-dialog"
        }
        
        live_model = model_mapping.get(model, "models/gemini-2.5-flash-live-preview")
        logger.debug(f"Using Live API model: {live_model}")
        
        # Connect and get response
        response_text = ""
        async with clients.live_client.aio.live.connect(model=live_model, config=config) as session:
            # Send the prompt
            await session.send(input=prompt, end_of_turn=True)
            
            # Receive response
            turn = session.receive()
            async for response in turn:
                if text := response.text:
                    response_text += text
        
        # Record successful request
        rate_limiter.record_request(model)
        
        if response_text.strip():
            return True, response_text.strip()
        else:
            return False, "No response from Live API"
            
    except Exception as e:
        # Check if it's a rate limit error
        retry_delay = await rate_limiter.handle_rate_limit_error(model, e)
        if retry_delay:
            await asyncio.sleep(retry_delay)
            return await call_gemini_live_api_async(messages, model, is_owner)
        
        logger.exception(f"Error calling Live API: {e}")
        return False, f"Live API error: {str(e)}"

async def call_gemini_live_api_stream_async(
    messages: List[Dict[str, str]], 
    model: str,
    is_owner: bool = False
) -> AsyncIterator[str]:
    """Stream response from Gemini Live API with tools support"""
    
    if not clients.live_available:
        yield clients.live_error or "Gemini Live API not available"
        return
    
    try:
        # Apply rate limiting
        await rate_limiter.wait_if_needed(model)
        
        # Get Live API tools
        tools = get_live_api_tools()
        
        # Create config with tools enabled
        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            media_resolution="MEDIA_RESOLUTION_LOW",
            tools=tools # Only include tools if they exist
        )
        
        # Convert messages to simple prompt
        prompt = build_gemini_prompt(messages)
        logger.debug(f"Live API stream prompt length: {len(prompt)} characters")
        logger.debug(f"Live API stream tools enabled: {len(tools) if tools else 0}")
        
        # Map model names to correct Live API model names
        model_mapping = {
            "gemini-2.5-flash-live-preview": "models/gemini-2.5-flash-live-preview",
            "gemini-2.5-flash-preview-native-audio-dialog": "models/gemini-2.5-flash-live-preview",
            "gemini-2.5-flash-exp-native-audio-thinking-dialog": "models/gemini-2.5-flash-live-preview"
        }
        
        live_model = model_mapping.get(model, "models/gemini-2.5-flash-live-preview")
        
        # Connect and stream response
        async with clients.live_client.aio.live.connect(model=live_model, config=config) as session:
            # Send the prompt
            await session.send(input=prompt, end_of_turn=True)
            
            # Stream response
            turn = session.receive()
            async for response in turn:
                if text := response.text:
                    yield text
        
        # Record successful request
        rate_limiter.record_request(model)
        
    except Exception as e:
        # Check if it's a rate limit error  
        retry_delay = await rate_limiter.handle_rate_limit_error(model, e)
        if retry_delay:
            await asyncio.sleep(retry_delay)
            async for chunk in call_gemini_live_api_stream_async(messages, model, is_owner):
                yield chunk
            return
        
        logger.exception(f"Error in Live API stream: {e}")
        yield f"Live API stream error: {str(e)}"

def call_gemini_api(
    messages: List[Dict[str, Any]], 
    model: str, 
    is_owner: bool = False,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    enable_tools: bool = True,
    enable_thinking: bool = True,
    thinking_budget: int = 25000
) -> Tuple[bool, str]:
    """Call Gemini API with integrated rate limiting"""
    try:
        # Apply rate limiting using the rate limiter
        asyncio.run(rate_limiter.wait_if_needed(model))
        
        # Configure API key
        if is_owner and clients.owner_gemini_available:
            genai.configure(api_key=load_config.OWNER_GEMINI_API_KEY)
            logger.debug("Using owner's Gemini API key")
        elif clients.gemini_available:
            genai.configure(api_key=load_config.CLIENT_GEMINI_API_KEY)
            logger.debug("Using regular Gemini API key")
        else:
            return False, clients.gemini_error or "Gemini API not initialized"

        # Check if we have multimodal content
        has_images = has_multimodal_content(messages)
        logger.debug(f"Multimodal content detected: {has_images}")

        # Try enhanced SDK branch if text-only and SDK available
        generate_content_config = None
        if not has_images:
            generate_content_config = create_gemini_generate_content_config(
                model=model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                enable_tools=enable_tools,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget
            )

        if generate_content_config and LIVE_API_AVAILABLE and not has_images:
            logger.debug("Using enhanced Gemini API with tools and thinking support")
            if is_owner and clients.owner_gemini_available:
                api_key = load_config.OWNER_GEMINI_API_KEY
            else:
                api_key = load_config.CLIENT_GEMINI_API_KEY
                
            enhanced_client = live_genai.Client(api_key=api_key)
            prompt = build_gemini_prompt(messages)
            logger.debug(f"Enhanced Gemini prompt length: {len(prompt)} characters")
            
            # Handle rate limiting with retries
            settings = rate_limiter.get_rate_limit(model)
            max_attempts = settings["retries"] + 1
            attempt = 0
            
            while attempt < max_attempts:
                try:
                    response = enhanced_client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=generate_content_config
                    )
                    break
                except Exception as e:
                    attempt += 1
                    retry_delay = asyncio.run(rate_limiter.handle_rate_limit_error(model, e))
                    
                    if retry_delay and attempt < max_attempts:
                        logger.warning(f"Rate limit hit, retrying after {retry_delay} seconds (attempt {attempt}/{max_attempts})")
                        asyncio.run(asyncio.sleep(retry_delay))
                    else:
                        raise e
                        
            if attempt >= max_attempts:
                return False, "Exceeded maximum retry attempts due to rate limits."
                
            # Record successful request
            rate_limiter.record_request(model)
            
            if hasattr(response, 'text') and response.text:
                return True, response.text.strip()
            else:
                return False, "No text content returned from enhanced API"
        else:
            # Use traditional API (supports images)
            logger.debug(f"Using traditional Gemini API (images: {has_images})")
            generation_config = create_generation_config(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens
            )
            logger.debug(f"Gemini generation config: {generation_config}")
            
            model_instance = genai.GenerativeModel(
                model,
                safety_settings=GeminiConfig.DEFAULT_SAFETY_SETTINGS,
                generation_config=generation_config
            )
            
            if has_images:
                prompt_content = build_gemini_prompt_multimodal(messages)
                logger.debug(f"Multimodal Gemini prompt parts: {len(prompt_content)}")
                image_count = 0
                for part in prompt_content:
                    if hasattr(part, 'size'):
                        image_count += 1
                    elif isinstance(part, dict) and part.get('mime_type'):
                        image_count += 1
                logger.info(f"Sending {image_count} images to Gemini model")
            else:
                prompt_content = build_gemini_prompt(messages)
                logger.debug(f"Text-only Gemini prompt length: {len(prompt_content)} characters")
            
            response = model_instance.generate_content(prompt_content)
            
            # Record successful request
            rate_limiter.record_request(model)
            
            return extract_gemini_response(response)
            
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["harm", "safety", "blocked"]):
            return False, "Response blocked by content safety filters. Please rephrase your request."
        logger.exception(f"Error calling Gemini API: {e}")
        return False, f"Gemini API error: {str(e)}"

async def call_gemini_api_stream(messages: List[Dict[str, Any]], model: str, is_owner: bool = False) -> AsyncIterator[str]:
    """Stream Gemini API calls with integrated rate limiting"""
    try:
        # Apply rate limiting
        await rate_limiter.wait_if_needed(model)
        
        # Configure API key
        if is_owner and clients.owner_gemini_available:
            genai.configure(api_key=load_config.OWNER_GEMINI_API_KEY)
        elif clients.gemini_available:
            genai.configure(api_key=load_config.CLIENT_GEMINI_API_KEY)
        else:
            yield "Gemini API not initialized"
            return
        
        # Check if we have multimodal content
        has_images = has_multimodal_content(messages)
        logger.debug(f"Streaming - Multimodal content detected: {has_images}")

        # Create model instance with streaming using original model name
        model_instance = genai.GenerativeModel(
            model_name=model,
            safety_settings=GeminiConfig.DEFAULT_SAFETY_SETTINGS
        )

        # Build appropriate prompt based on content type
        if has_images:
            # Multimodal content - use new format
            prompt_content = build_gemini_prompt_multimodal(messages)
            logger.debug(f"Streaming - Multimodal Gemini prompt parts: {len(prompt_content)}")
            
            # Log image information
            image_count = 0
            for part in prompt_content:
                if hasattr(part, 'size'):  # PIL Image
                    image_count += 1
                elif isinstance(part, dict) and part.get('mime_type'):
                    image_count += 1
                    
            logger.info(f"Streaming {image_count} images to Gemini model")
        else:
            # Text-only content
            prompt_content = build_gemini_prompt(messages)
            logger.debug(f"Streaming - Text-only Gemini prompt length: {len(prompt_content)} characters")
        
        # Get streaming response
        response = model_instance.generate_content(prompt_content, stream=True)
        
        # Record successful request
        rate_limiter.record_request(model)
        
        # Convert sync iterator to async
        for chunk in response:
            if chunk.text:
                yield chunk.text
            await asyncio.sleep(0)  # Allow other coroutines to run

    except Exception as e:
        # Handle rate limit errors
        retry_delay = await rate_limiter.handle_rate_limit_error(model, e)
        if retry_delay:
            await asyncio.sleep(retry_delay)
            async for chunk in call_gemini_api_stream(messages, model, is_owner):
                yield chunk
            return
            
        logger.exception(f"Error in Gemini stream: {e}")
        yield f"Stream error: {str(e)}"

def is_model_available(model: str) -> Tuple[bool, str]:
    """Check if a model is available for use"""
    try:
        # Get MongoDB store instance
        _mongodb_store = get_mongodb_store()
        
        # Check profile model first
        if _mongodb_store and _mongodb_store.get_profile_model(model):
            return True, ""

        if is_gemini_model(model):
            # Check Live API models
            if is_gemini_live_model(model):
                if not clients.live_available:
                    return False, clients.live_error or "Gemini Live API is not available"
                return True, ""
            
            # Regular Gemini models
            if not clients.gemini_available:
                return False, clients.gemini_error or "Gemini API is not available"
            return True, ""
        return True, ""
        
    except Exception as e:
        logger.error(f"Error checking model availability: {e}")
        return False, str(e)

def call_openai_proxy(
    messages: List[Dict[str, Any]], 
    model: str = "gpt-3.5-turbo", 
    is_owner: bool = False,
    # Gemini-specific parameters (ignored for OpenAI models)
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    # New Gemini-specific parameters for tools and thinking
    enable_tools: bool = True,
    enable_thinking: bool = True,
    thinking_budget: int = 16000
) -> Tuple[bool, str]:
    """
    Call OpenAI API or Gemini API based on model type
    
    Args:
        messages: List of message dictionaries (can include multimodal content)
        model: Model name
        is_owner: Whether to use owner privileges
        temperature: Temperature for generation
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter (Gemini only)  
        max_output_tokens: Maximum output tokens
        enable_tools: Enable tools for Gemini models
        enable_thinking: Enable thinking for supported Gemini models
        thinking_budget: Thinking budget for reasoning models
    
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
                max_output_tokens=max_output_tokens,
                enable_tools=enable_tools,
                enable_thinking=enable_thinking,
                thinking_budget=thinking_budget
            )
        else:
            # OpenAI API call (Gemini-specific parameters ignored)
            # Convert multimodal messages to text-only for OpenAI
            openai_messages = []
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Extract only text parts for OpenAI
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    text_content = " ".join(text_parts)
                    openai_messages.append({
                        "role": msg.get("role", "user"),
                        "content": text_content
                    })
                else:
                    openai_messages.append(msg)
            
            openai_temperature = temperature if temperature is not None else 1.1
            
            response = clients.openai_client.chat.completions.create(
                model=model,
                messages=openai_messages,
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

async def call_openai_proxy_stream(
    messages: List[Dict[str, Any]],
    model: str = "gpt-3.5-turbo",
    is_owner: bool = False
) -> AsyncIterator[str]:
    """Unified streaming API call handler with IMAGE SUPPORT and integrated rate limiting"""

    available, error = is_model_available(model)
    if not available:
        yield error
        return
        
    if is_gemini_live_model(model):
        async for chunk in call_gemini_live_api_stream_async(messages, model, is_owner):
            yield chunk
    elif is_gemini_model(model):
        # Use regular Gemini streaming with image support
        async for chunk in call_gemini_api_stream(messages, model, is_owner):
            yield chunk
    else:
        # Regular non-streaming call for OpenAI (convert multimodal to text-only)
        openai_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Extract only text parts for OpenAI
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                text_content = " ".join(text_parts)
                openai_messages.append({
                    "role": msg.get("role", "user"),
                    "content": text_content
                })
            else:
                openai_messages.append(msg)
        
        ok, response = call_openai_proxy(openai_messages, model, is_owner)
        if ok:
            yield response
        else:
            yield f"Error: {response}"