# call_api.py
import logging
import os
from openai import OpenAI
from google import genai
import load_config  # changed from relative import to absolute import

logger = logging.getLogger("discord-openai-proxy.call_api")

# Initialize OpenAI client (proxy support)
try:
    openai_client = OpenAI(api_key=load_config.OPENAI_API_KEY, base_url=load_config.OPENAI_API_BASE)
except TypeError:
    # fallback constructor if SDK signature differs
    openai_client = OpenAI(api_key=load_config.OPENAI_API_KEY)

# Initialize Gemini client
gemini_client = None
GEMINI_AVAILABLE = False  # New flag to track availability
GEMINI_ERROR = None  # Store error message

try:
    # First check if key exists
    if not hasattr(load_config, 'GEMINI_API_KEY') or not load_config.GEMINI_API_KEY:
        GEMINI_ERROR = "Gemini API key not found in config"
        logger.warning(GEMINI_ERROR)
    else:
        try:
            # Then try to import and initialize
            from google import genai
            os.environ['GEMINI_API_KEY'] = load_config.GEMINI_API_KEY
            gemini_client = genai.Client()
            GEMINI_AVAILABLE = True
            logger.info("Gemini client initialized successfully")
        except ImportError:
            GEMINI_ERROR = "Google GenAI library not installed. Please run: pip install google-generativeai"
            logger.error(GEMINI_ERROR)
        except Exception as e:
            GEMINI_ERROR = f"Error initializing Gemini client: {e}"
            logger.error(GEMINI_ERROR)
except Exception as e:
    GEMINI_ERROR = f"Unexpected error checking Gemini configuration: {e}"
    logger.error(GEMINI_ERROR)


def is_gemini_model(model_name: str) -> bool:
    """Check if the model is a Gemini model"""
    return model_name.startswith("gemini-")


def convert_messages_to_gemini_format(messages):
    """Convert OpenAI format messages to Gemini format"""
    try:
        # Gemini expects a single content string, so we'll combine all messages
        # Skip system messages or incorporate them into the user content
        user_messages = []
        system_content = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_content = content
            elif role == "user":
                if system_content:
                    user_messages.append(f"System: {system_content}\n\nUser: {content}")
                    system_content = ""  # Only use system prompt once
                else:
                    user_messages.append(content)
            elif role == "assistant":
                user_messages.append(f"Assistant: {content}")

        # Combine all messages into a single content string
        combined_content = "\n\n".join(user_messages)
        return combined_content

    except Exception as e:
        logger.exception("Error converting messages to Gemini format")
        # Fallback - just use the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return "Hello"


def call_gemini_api(messages, model):
    """Call Gemini API with the given messages and model"""
    if not gemini_client:
        return False, "Gemini client not initialized"

    try:
        # Convert OpenAI format to Gemini format
        content = convert_messages_to_gemini_format(messages)

        # Call Gemini API
        response = gemini_client.models.generate_content(
            model=model,
            contents=content
        )

        # Extract the response text
        response_text = response.text if hasattr(response, 'text') else str(response)
        return True, response_text

    except Exception as e:
        logger.exception(f"Error calling Gemini API: {e}")
        return False, str(e)


def is_model_available(model: str) -> tuple[bool, str]:
    """Check if a model is available for use"""
    if is_gemini_model(model):
        if not GEMINI_AVAILABLE:
            return False, GEMINI_ERROR or "Gemini API is not available"
        return True, ""
    return True, ""  # Non-Gemini models are assumed available


def call_openai_proxy(messages, model="gpt-3.5-turbo"):
    """
    Call OpenAI API or Gemini API based on model type
    """
    try:
        # Check model availability first
        available, error = is_model_available(model)
        if not available:
            return False, error

        if is_gemini_model(model):
            # Let convert_messages_to_gemini_format handle the conversion
            return call_gemini_api(messages, model)
        else:
            # Use standard OpenAI format
            response = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.1,
                timeout=load_config.REQUEST_TIMEOUT
            )

            choice = response.choices[0]
            if choice.finish_reason == "length":
                logger.warning(f"Response truncated due to max_tokens limit for model {model}")

            return True, choice.message.content

    except Exception as e:
        logger.exception(f"Error calling API for model {model}: {e}")
        return False, str(e)