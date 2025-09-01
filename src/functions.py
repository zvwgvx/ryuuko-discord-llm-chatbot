#!/usr/bin/env python3
# coding: utf-8
# ────────────────────────────────────────────────────────────────────────────
# Bot helper / command registry - UPDATED FOR SLASH COMMANDS WITH IMAGE SUPPORT
# Uses MemoryStore for per‑user conversation history
# Uses UserConfigManager for per-user model and system prompt settings
# Uses MongoDB for model management
# ────────────────────────────────────────────────────────────────────────────

import re
import json
import logging
import asyncio
import time
import base64
import mimetypes
from pathlib import Path
from typing import Set, Optional, List, Dict, Union
from datetime import datetime, timezone, timedelta

import discord
from discord.ext import commands
from discord import app_commands

# ────────────────────────────────────────────────────────────────────────────
# ***Absolute import — no package, so we use the plain module name ***
from memory_store import MemoryStore
from user_config import get_user_config_manager
from request_queue import get_request_queue

logger = logging.getLogger("functions")

# ---------------------------- module‑level state -----------------------------
_bot: Optional[commands.Bot] = None
_call_api = None
_config = None
_user_config_manager = None
_request_queue = None

# ---------------------------------------------------------------
# Persistence helpers — authorized user IDs
# ---------------------------------------------------------------
_authorized_users: Set[int] = set()

# MongoDB storage globals
_use_mongodb_auth = False
_mongodb_store = None

# ---------------------------------------------------------------
# Attachment handling constants - UPDATED FOR IMAGES
# ---------------------------------------------------------------
FILE_MAX_BYTES = 200 * 1024          # 200 KB per file
IMAGE_MAX_BYTES = 10 * 1024 * 1024   # 10 MB per image
MAX_CHARS_PER_FILE = 10_000
ALLOWED_TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".java", ".c", ".cpp", ".h",
    ".json", ".yaml", ".yml", ".csv", ".rs", ".go", ".rb",
    ".sh", ".html", ".css", ".ts", ".ini", ".toml",
}
ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"
}
ALLOWED_IMAGE_MIMES = {
    "image/jpeg", "image/jpg", "image/png", "image/gif", 
    "image/webp", "image/bmp"
}

# ---------------------------------------------------------------
# Optional memory store
# ---------------------------------------------------------------
_memory_store: Optional[MemoryStore] = None

# ------------------------------------------------------------------
# Persistence helpers — authorized users
# ------------------------------------------------------------------

def load_authorized_from_path(path: Path) -> Set[int]:
    """Load authorized users from file (legacy mode)"""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            arr = data.get("authorized", [])
            return set(int(x) for x in arr)
        except Exception:
            logger.exception("Failed to load authorized.json, returning empty set.")
    return set()

def save_authorized_to_path(path: Path, s: Set[int]) -> None:
    """Save authorized users to file (legacy mode)"""
    try:
        path.write_text(json.dumps({"authorized": sorted(list(s))}, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to save authorized.json")

def load_authorized_users() -> Set[int]:
    """Load authorized users from storage backend"""
    global _use_mongodb_auth, _mongodb_store
    
    if _use_mongodb_auth and _mongodb_store:
        return _mongodb_store.get_authorized_users()
    else:
        return load_authorized_from_path(_config.AUTHORIZED_STORE)
    
def add_authorized_user(user_id: int) -> bool:
    """Add user to authorized list"""
    global _authorized_users, _use_mongodb_auth, _mongodb_store
    
    if _use_mongodb_auth and _mongodb_store:
        success = _mongodb_store.add_authorized_user(user_id)
        if success:
            _authorized_users.add(user_id)
        return success
    else:
        _authorized_users.add(user_id)
        save_authorized_to_path(_config.AUTHORIZED_STORE, _authorized_users)
        return True
    
def remove_authorized_user(user_id: int) -> bool:
    """Remove user from authorized list"""
    global _authorized_users, _use_mongodb_auth, _mongodb_store
    
    if _use_mongodb_auth and _mongodb_store:
        success = _mongodb_store.remove_authorized_user(user_id)
        if success:
            _authorized_users.discard(user_id)
        return success
    else:
        if user_id in _authorized_users:
            _authorized_users.remove(user_id)
            save_authorized_to_path(_config.AUTHORIZED_STORE, _authorized_users)
            return True
        return False

# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------
async def is_authorized_user(user: discord.abc.User) -> bool:
    """Return True if `user` is the bot owner or in the authorized set."""
    global _bot, _authorized_users
    try:
        if await _bot.is_owner(user):
            return True
    except Exception:
        pass
    return getattr(user, "id", None) in _authorized_users

def _extract_user_id_from_str(s: str) -> Optional[int]:
    m = re.search(r"(\d{17,20})", s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return None
    return None

def should_respond_default(message: discord.Message) -> bool:
    """Return True for a DM or an explicit mention of the bot."""
    if isinstance(message.channel, discord.DMChannel):
        return True
    if _bot.user in message.mentions:
        return True
    return False

def is_gemini_model(model_name: str) -> bool:
    """Check if the model is a Gemini model"""
    return (model_name.startswith("gemini-") or 
            model_name.startswith("gemma-") or 
            "live-preview" in model_name)

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

# ------------------------------------------------------------------
# Enhanced Attachment helpers - WITH IMAGE SUPPORT
# ------------------------------------------------------------------

async def _read_image_attachment(attachment: discord.Attachment) -> Dict:
    """Process an image attachment for Gemini API"""
    entry = {
        "filename": attachment.filename,
        "type": "image",
        "data": None,
        "mime_type": None,
        "skipped": False,
        "reason": None
    }

    try:
        # Size check
        size = getattr(attachment, "size", 0) or 0
        if size > IMAGE_MAX_BYTES:
            entry["skipped"] = True
            entry["reason"] = f"image too large ({size} bytes, max {IMAGE_MAX_BYTES})"
            return entry

        # Content type check
        content_type = getattr(attachment, "content_type", "") or ""
        ext = (Path(attachment.filename).suffix or "").lower()
        
        if not (content_type in ALLOWED_IMAGE_MIMES or ext in ALLOWED_IMAGE_EXTENSIONS):
            entry["skipped"] = True
            entry["reason"] = f"unsupported image type ({content_type}, {ext})"
            return entry

        # Read image data
        image_data = await attachment.read()
        
        # Convert to base64
        base64_data = base64.b64encode(image_data).decode('utf-8')
        
        # Determine MIME type
        mime_type = content_type if content_type in ALLOWED_IMAGE_MIMES else mimetypes.guess_type(attachment.filename)[0]
        if not mime_type:
            # Fallback based on extension
            mime_mapping = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif",
                ".webp": "image/webp", ".bmp": "image/bmp"
            }
            mime_type = mime_mapping.get(ext, "image/jpeg")

        entry["data"] = base64_data
        entry["mime_type"] = mime_type
        logger.info(f"Successfully processed image: {attachment.filename} ({size} bytes, {mime_type})")

    except Exception as e:
        logger.exception(f"Error reading image attachment {attachment.filename}")
        entry["skipped"] = True
        entry["reason"] = f"read error: {e}"

    return entry

async def _read_text_attachment(attachment: discord.Attachment) -> Dict:
    """Process a text attachment"""
    entry = {"filename": attachment.filename, "type": "text", "text": "", "skipped": False, "reason": None}

    # quick size check
    try:
        size = int(getattr(attachment, "size", 0) or 0)
    except Exception:
        size = 0

    ext = (Path(attachment.filename).suffix or "").lower()
    content_type = getattr(attachment, "content_type", "") or ""

    # filter by content‑type / extension
    if not (
        content_type.startswith("text")
        or content_type in ("application/json", "application/javascript")
        or ext in ALLOWED_TEXT_EXTENSIONS
    ):
        entry["skipped"] = True
        entry["reason"] = f"unsupported file type ({content_type!r}, {ext!r})"
        return entry

    if size and size > FILE_MAX_BYTES:
        entry["skipped"] = True
        entry["reason"] = f"file too large ({size} bytes)"
        return entry

    try:
        b = await attachment.read()
        try:
            text = b.decode("utf-8")
        except Exception:
            try:
                text = b.decode("latin-1")
            except Exception:
                text = b.decode("utf-8", errors="replace")

        # truncate very long files
        if len(text) > MAX_CHARS_PER_FILE:
            text = text[:MAX_CHARS_PER_FILE] + "\n\n...[truncated]..."

        entry["text"] = text
    except Exception as e:
        logger.exception("Error reading attachment %s", attachment.filename)
        entry["skipped"] = True
        entry["reason"] = f"read error: {e}"

    return entry

async def _read_attachments_enhanced(attachments: List[discord.Attachment]) -> Dict:
    """Enhanced attachment processing with image support"""
    result = {
        "text_files": [],
        "images": [],
        "text_summary": "",
        "has_images": False
    }
    
    for att in attachments:
        ext = (Path(att.filename).suffix or "").lower()
        content_type = getattr(att, "content_type", "") or ""
        
        # Determine if this is an image
        if (content_type in ALLOWED_IMAGE_MIMES or ext in ALLOWED_IMAGE_EXTENSIONS):
            # Process as image
            image_entry = await _read_image_attachment(att)
            result["images"].append(image_entry)
            if not image_entry["skipped"]:
                result["has_images"] = True
        else:
            # Process as text file
            text_entry = await _read_text_attachment(att)
            result["text_files"].append(text_entry)

    # Build text summary for text files
    attach_summary = []
    files_combined = ""
    
    for fi in result["text_files"]:
        if fi.get("skipped"):
            attach_summary.append(f"- {fi['filename']}: SKIPPED ({fi.get('reason')})")
        else:
            attach_summary.append(f"- {fi['filename']}: included ({len(fi['text'])} chars)")
            files_combined += f"Filename: {fi['filename']}\n---\n{fi['text']}\n\n"
    
    # Add image summary
    for img in result["images"]:
        if img.get("skipped"):
            attach_summary.append(f"- {img['filename']}: SKIPPED ({img.get('reason')})")
        else:
            attach_summary.append(f"- {img['filename']}: image included ({img.get('mime_type')})")

    if attach_summary:
        result["text_summary"] = "\n".join(attach_summary) + "\n\n" + files_combined

    return result

# ------------------------------------------------------------------
# Message formatting helpers with enhanced long message handling
# ------------------------------------------------------------------

def convert_latex_to_discord(text: str) -> str:
    """Fixed version - only protect code blocks, NOT markdown tables"""
    
    # Step 1: Only protect code regions, NOT tables
    protected_regions = []
    
    def protect_region(match):
        content = match.group(0)
        placeholder = f"__PROTECTED_{len(protected_regions)}__"
        protected_regions.append(content)
        return placeholder
    
    # Only protect code-related patterns - DO NOT protect tables
    patterns_to_protect = [
        r'```[\s\S]*?```',  # Code blocks
        r'`[^`\n]*?`',      # Inline code only
        # Programming patterns (but not tables!)
        r'#include\s*<[^>]+>', # C++ includes
        r'\b(?:cout|cin|std::)\b[^.\n]*?;',  # C++ statements
        r'\bfor\s*\([^)]*\)\s*\{[^}]*\}',   # For loops
        r'\bwhile\s*\([^)]*\)\s*\{[^}]*\}', # While loops
        r'\bif\s*\([^)]*\)\s*\{[^}]*\}',    # If statements
    ]
    
    working_text = text
    for pattern in patterns_to_protect:
        working_text = re.sub(pattern, protect_region, working_text, flags=re.MULTILINE | re.DOTALL)
    
    # Step 2: Apply LaTeX conversion to remaining text (including tables)
    # Simple replacements for common LaTeX symbols
    latex_replacements = {
        r'\\cdot\b': '·', r'\\times\b': '×', r'\\div\b': '÷', r'\\pm\b': '±',
        r'\\leq\b': '≤', r'\\geq\b': '≥', r'\\neq\b': '≠', r'\\approx\b': '≈',
        r'\\alpha\b': 'α', r'\\beta\b': 'β', r'\\gamma\b': 'γ', r'\\delta\b': 'δ',
        r'\\pi\b': 'π', r'\\sigma\b': 'σ', r'\\lambda\b': 'λ', r'\\mu\b': 'μ',
        r'\\rightarrow\b': '→', r'\\to\b': '→', r'\\leftarrow\b': '←',
        r'\\sum\b': 'Σ', r'\\prod\b': 'Π', r'\\int\b': '∫',
        r'\\infty\b': '∞', r'\\emptyset\b': '∅',
    }
    
    for latex_pattern, replacement in latex_replacements.items():
        working_text = re.sub(latex_pattern, replacement, working_text)
    
    # Handle fractions \frac{a}{b} -> a/b
    def replace_fraction(match):
        numerator = match.group(1).strip()
        denominator = match.group(2).strip()
        if len(numerator) <= 3 and len(denominator) <= 3:
            return f'{numerator}/{denominator}'
        else:
            return f'({numerator})/({denominator})'
    
    working_text = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', replace_fraction, working_text)
    
    # Step 3: Restore protected regions
    for i, protected_content in enumerate(protected_regions):
        placeholder = f"__PROTECTED_{i}__"
        working_text = working_text.replace(placeholder, protected_content)
    
    return working_text

def is_table_line(line: str) -> bool:
    """Check if a line is part of a markdown table"""
    stripped = line.strip()
    # Table data line: starts and ends with |, has at least 2 |
    if stripped.startswith('|') and stripped.endswith('|') and stripped.count('|') >= 2:
        return True
    # Table separator line: |---|---| or |:---|---:| etc
    if (stripped.startswith('|') and 
        all(c in '|-: \t' for c in stripped) and
        '-' in stripped and stripped.count('|') >= 2):
        return True
    return False

def find_complete_table(lines: list, start_idx: int) -> tuple:
    """Find the complete table boundaries starting from any table line"""
    if start_idx >= len(lines) or not is_table_line(lines[start_idx]):
        return start_idx, start_idx
    
    # Find table start (go backwards)
    table_start = start_idx
    while table_start > 0 and is_table_line(lines[table_start - 1]):
        table_start -= 1
    
    # Find table end (go forwards)
    table_end = start_idx
    while table_end < len(lines) - 1 and is_table_line(lines[table_end + 1]):
        table_end += 1
    
    return table_start, table_end

def handle_large_table(table_lines: list, max_length: int, chunks: list, current_chunk: str) -> str:
    """Handle tables that are too large to fit in one message"""
    # Try to identify header vs data
    header_lines = []
    data_lines = []
    
    # Usually first 2 lines are header + separator
    for j, tline in enumerate(table_lines):
        if j < 2:
            header_lines.append(tline)
        else:
            data_lines.append(tline)
    
    if len(header_lines) >= 2:
        header_text = '\n'.join(header_lines)
        if len(header_text) <= max_length:
            # Start with header
            current_table_chunk = header_text
            
            # Add data lines one by one
            for data_line in data_lines:
                test_line = current_table_chunk + '\n' + data_line
                if len(test_line) <= max_length:
                    current_table_chunk = test_line
                else:
                    # Current chunk is full, save it
                    chunks.append(current_table_chunk)
                    # Start new chunk with header + current data line
                    current_table_chunk = header_text + '\n' + data_line
            
            return current_table_chunk
        else:
            # Even header is too long, fallback to line by line
            result_chunk = current_chunk
            for tline in table_lines:
                test_line = result_chunk + ('\n' if result_chunk else '') + tline
                if len(test_line) <= max_length:
                    result_chunk = test_line
                else:
                    if result_chunk:
                        chunks.append(result_chunk)
                    result_chunk = tline
            return result_chunk
    else:
        # Fallback: process line by line
        result_chunk = current_chunk
        for tline in table_lines:
            test_line = result_chunk + ('\n' if result_chunk else '') + tline
            if len(test_line) <= max_length:
                result_chunk = test_line
            else:
                if result_chunk:
                    chunks.append(result_chunk)
                result_chunk = tline
        return result_chunk

def handle_long_line(line: str, max_length: int, chunks: list) -> str:
    """Handle individual lines that are too long"""
    if len(line) <= max_length:
        return line
        
    # Preserve indentation
    leading_whitespace = re.match(r'^(\s*)', line).group(1)
    line_content = line[len(leading_whitespace):]
    
    # Try to split at natural break points
    current_content = ""
    words = line_content.split()
    
    for word in words:
        test_line = leading_whitespace + current_content + (' ' if current_content else '') + word
        
        if len(test_line) <= max_length:
            current_content += (' ' if current_content else '') + word
        else:
            if current_content:
                chunks.append(leading_whitespace + current_content)
                current_content = word
            else:
                # Single word is too long - force split
                while len(word) > max_length - len(leading_whitespace):
                    available_space = max_length - len(leading_whitespace)
                    chunks.append(leading_whitespace + word[:available_space])
                    word = word[available_space:]
                current_content = word
    
    return leading_whitespace + current_content if current_content else ""

def split_message_smart(text: str, max_length: int = 2000) -> list[str]:
    """Enhanced smart message splitting that preserves tables, code blocks, and context"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_chunk = ""
    in_code_block = False
    code_block_lang = ""
    
    lines = text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Handle code blocks
        code_match = re.match(r'^```(\w*)', line.strip())
        if code_match:
            if not in_code_block:
                in_code_block = True
                code_block_lang = code_match.group(1)
            elif line.strip() == '```':
                in_code_block = False
                code_block_lang = ""
        
        # Handle tables (only when NOT in code block)
        if is_table_line(line) and not in_code_block:
            table_start, table_end = find_complete_table(lines, i)
            
            # Get the entire table as one unit
            table_lines = lines[table_start:table_end + 1]
            table_text = '\n'.join(table_lines)
            
            # Try to add the complete table to current chunk
            test_chunk = current_chunk + ('\n' if current_chunk else '') + table_text
            
            if len(test_chunk) <= max_length:
                # Table fits in current chunk
                current_chunk = test_chunk
            else:
                # Table doesn't fit
                if current_chunk:
                    # Save current chunk first
                    if in_code_block:
                        current_chunk += '\n```'
                    chunks.append(current_chunk)
                    if in_code_block:
                        current_chunk = f'```{code_block_lang}'
                    else:
                        current_chunk = ""
                
                # Handle the table
                if len(table_text) <= max_length:
                    # Table fits in its own chunk
                    current_chunk = table_text
                else:
                    # Table is too large - split more intelligently
                    current_chunk = handle_large_table(table_lines, max_length, chunks, current_chunk)
            
            # Skip to after the table
            i = table_end + 1
            continue
        
        # Regular line processing (not part of a table)
        test_chunk = current_chunk + ('\n' if current_chunk else '') + line
        
        if len(test_chunk) > max_length:
            if current_chunk:
                # Save current chunk
                if in_code_block:
                    current_chunk += '\n```'
                    chunks.append(current_chunk)
                    current_chunk = f'```{code_block_lang}\n{line}'
                else:
                    chunks.append(current_chunk)
                    current_chunk = line
            else:
                # Single line is too long - split it preserving structure
                current_chunk = handle_long_line(line, max_length, chunks)
        else:
            current_chunk = test_chunk
        
        i += 1
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Post-process chunks to ensure no empty chunks and proper formatting
    final_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if chunk:
            final_chunks.append(chunk)
    
    return final_chunks if final_chunks else ["[Empty response]"]

async def send_long_message(channel, content: str, max_msg_length: int = 2000):
    """Send long message with proper table handling"""
    try:
        # First apply LaTeX conversion (which now preserves tables)
        formatted_content = convert_latex_to_discord(content)
        
        if len(formatted_content) <= max_msg_length:
            await channel.send(formatted_content, allowed_mentions=discord.AllowedMentions.none())
            return
        
        chunks = split_message_smart(formatted_content, max_msg_length)
        
        for i, chunk in enumerate(chunks):
            if i > 0:  # Add delay between messages
                await asyncio.sleep(0.5)
            await channel.send(chunk, allowed_mentions=discord.AllowedMentions.none())
    except Exception as e:
        logger.exception("Error in send_long_message")
        try:
            await channel.send(f"Error sending message: {str(e)[:100]}", 
                             allowed_mentions=discord.AllowedMentions.none())
        except:
            pass

async def send_long_message_with_reference(channel, content: str, reference_message: discord.Message, max_msg_length: int = 2000):
    """Enhanced version with better error handling and retry logic"""
    try:
        # First apply LaTeX conversion (which now preserves tables)
        formatted_content = convert_latex_to_discord(content)
        
        if len(formatted_content) <= max_msg_length:
            await channel.send(
                formatted_content,
                reference=reference_message,
                allowed_mentions=discord.AllowedMentions.none()
            )
            return
        
        chunks = split_message_smart(formatted_content, max_msg_length)
        
        for i, chunk in enumerate(chunks):
            try:
                if i > 0:  # Add delay between messages
                    await asyncio.sleep(0.5)  # Slightly longer delay
                
                # Only reference the original message for the first chunk
                ref = reference_message if i == 0 else None
                await channel.send(
                    chunk,
                    reference=ref,
                    allowed_mentions=discord.AllowedMentions.none()
                )
            except discord.errors.HTTPException as e:
                if "Must be 2000 or fewer in length" in str(e):
                    # Chunk is still too long - force split further
                    logger.warning(f"Chunk {i} still too long ({len(chunk)} chars), force splitting")
                    mini_chunks = split_message_smart(chunk, max_msg_length - 100)  # Leave buffer
                    for j, mini_chunk in enumerate(mini_chunks):
                        await asyncio.sleep(0.3)
                        ref = reference_message if i == 0 and j == 0 else None
                        await channel.send(
                            mini_chunk,
                            reference=ref,
                            allowed_mentions=discord.AllowedMentions.none()
                        )
                else:
                    logger.error(f"Failed to send chunk {i}: {e}")
                    # Try to send error message
                    try:
                        await channel.send(
                            f"Error sending part {i+1} of response: {str(e)[:100]}",
                            allowed_mentions=discord.AllowedMentions.none()
                        )
                    except:
                        pass
            except Exception as e:
                logger.exception(f"Unexpected error sending chunk {i}")
                
    except Exception as e:
        logger.exception("Critical error in send_long_message_with_reference")
        try:
            await channel.send(
                f"Critical error sending response: {str(e)[:100]}",
                reference=reference_message,
                allowed_mentions=discord.AllowedMentions.none()
            )
        except:
            pass

# ------------------------------------------------------------------
# AI Request Processing Function with Enhanced Image Support
# ------------------------------------------------------------------

def get_vietnam_timestamp() -> str:
    """Get current timestamp in GMT+7 (Vietnam timezone)"""
    vietnam_tz = timezone(timedelta(hours=7))
    now = datetime.now(vietnam_tz)
    
    formatted_time = now.strftime("%A, %B %d, %Y - %H:%M:%S")
    return f"Current time: {formatted_time} (GMT+7) : "

async def deduct_credits(user_id: int, amount: int) -> bool:
    """Deduct credits from user balance"""
    if not _use_mongodb_auth or not _mongodb_store:
        return True
        
    success, remaining = _mongodb_store.deduct_user_credit(user_id, amount)
    if not success:
        logger.error(f"Failed to deduct {amount} credits from user {user_id}")
    else:
        logger.info(f"Deducted {amount} credits from user {user_id}. Remaining: {remaining}")
    return success

async def check_model_access(message, model_info, user_id) -> bool:
    """Check if user has access to model"""
    if not model_info:
        return True
        
    # Check level
    user_config = _user_config_manager.get_user_config(user_id)
    user_level = user_config.get("access_level", 0)
    required_level = model_info.get("access_level", 0)
    
    if user_level < required_level:
        await message.channel.send(
            f"⛔ This model requires access level {required_level}. Your level: {user_level}",
            reference=message,
            allowed_mentions=discord.AllowedMentions.none()
        )
        return False
        
    # Check credits
    cost = model_info.get("credit_cost", 0)
    if cost > 0:
        current_credit = user_config.get("credit", 0)
        if current_credit < cost:
            await message.channel.send(
                f"⛔ Insufficient credits. This model costs {cost} credits per use. Your balance: {current_credit}",
                reference=message,
                allowed_mentions=discord.AllowedMentions.none()
            )
            return False
            
    return True

async def build_message_payload_with_images(user_id: int, final_user_text: str, images: List[Dict]) -> List[Dict]:
    """Build message payload with image support for Gemini models"""
    payload_messages = []
    
    # Add system message
    user_system_message = _user_config_manager.get_user_system_message(user_id)
    payload_messages.append(user_system_message)
    
    # Add conversation history if available
    if _memory_store:
        payload_messages.extend(_memory_store.get_user_messages(user_id))
    
    # Prepare final user message with images
    final_message_content = []
    
    # Add text content
    if final_user_text:
        final_message_content.append({
            "type": "text",
            "text": f"{get_vietnam_timestamp()}{final_user_text}" if _memory_store else final_user_text
        })
    
    # Add images for Gemini models
    for img in images:
        if not img.get("skipped") and img.get("data"):
            final_message_content.append({
                "type": "image",
                "image_data": {
                    "data": img["data"],
                    "mime_type": img["mime_type"]
                }
            })
            logger.info(f"Added image to payload: {img['filename']} ({img['mime_type']})")
    
    # Create final user message
    if len(final_message_content) == 1 and final_message_content[0]["type"] == "text":
        # Simple text-only message
        payload_messages.append({
            "role": "user", 
            "content": final_message_content[0]["text"]
        })
    else:
        # Multi-modal message with images
        payload_messages.append({
            "role": "user",
            "content": final_message_content
        })
    
    return payload_messages

async def process_ai_request(request):
    """Process a single AI request from the queue with enhanced image support"""
    message = request.message
    final_user_text = request.final_user_text
    user_id = message.author.id
    
    try:
        # Get user configuration
        user_model = _user_config_manager.get_user_model(user_id)
        
        # Check if using a profile model
        profile = None
        if _use_mongodb_auth:
            profile = _mongodb_store.get_profile_model(user_model)
        
        if profile:
            # Use profile settings
            user_model = profile["base_model"]
            user_system_message = {"role": "system", "content": profile["sys_prompt"]}
            is_live = profile.get("is_live", False)
            
            # Check access level and credits
            model_info = {"credit_cost": profile["credit_cost"], "access_level": profile["access_level"]}
            if not await check_model_access(message, model_info, user_id):
                return
        else:
            # Use regular user settings
            user_system_message = _user_config_manager.get_user_system_message(user_id)
            is_live = "live-preview" in user_model
            
            if _use_mongodb_auth:
                model_info = _mongodb_store.get_model_info(user_model)
                if not await check_model_access(message, model_info, user_id):
                    return

        # Check for images and model compatibility
        attachments = list(message.attachments or [])
        attachment_data = await _read_attachments_enhanced(attachments)
        has_images = attachment_data["has_images"]
        
        # Validate image support
        if has_images:
            if not is_gemini_model(user_model):
                await message.channel.send(
                    "🖼️ Images are only supported with Gemini models. Please switch to a Gemini model to use image analysis.",
                    reference=message,
                    allowed_mentions=discord.AllowedMentions.none()
                )
                return
            
            if is_gemini_live_model(user_model):
                await message.channel.send(
                    "🖼️ Images are not supported with Gemini Live models. Please switch to a regular Gemini model for image analysis.",
                    reference=message,
                    allowed_mentions=discord.AllowedMentions.none()
                )
                return
            
            logger.info(f"Processing {len(attachment_data['images'])} images with Gemini model: {user_model}")

        # Build final user text with attachments
        combined_text = ""
        if attachment_data["text_summary"]:
            combined_text += attachment_data["text_summary"]
        if final_user_text:
            combined_text += final_user_text
        
        if not combined_text.strip() and not has_images:
            await message.channel.send(
                "Please send a message with your question or attach some files.",
                reference=message,
                allowed_mentions=discord.AllowedMentions.none()
            )
            return

        # Build message payload
        if has_images and is_gemini_model(user_model) and not is_gemini_live_model(user_model):
            # Use enhanced payload with images for compatible Gemini models
            payload_messages = await build_message_payload_with_images(
                user_id, combined_text, attachment_data["images"]
            )
        else:
            # Standard text-only payload
            payload_messages = [user_system_message]
            if _memory_store:
                payload_messages.extend(_memory_store.get_user_messages(user_id))
                
                final = f"{get_vietnam_timestamp()}{combined_text}"
            payload_messages.append({"role": "user", "content": final})

        # Stream or regular response
        if is_live:
            # STREAMING RESPONSE - Enhanced version with proper long message handling
            collected_response = ""
            last_update = 0
            response_msg = None
            message_chunks = []  # Track multiple messages for long responses
            
            # Process stream
            async for chunk in _call_api.call_openai_proxy_stream(payload_messages, user_model):
                collected_response += chunk
                
                # Update message every 1 second or when chunk is large
                current_time = time.time()
                if (current_time - last_update > 1 or len(chunk) > 100) and collected_response:
                    formatted_response = convert_latex_to_discord(collected_response)
                    
                    try:
                        # Check if current response fits in single message
                        if len(formatted_response) <= 2000:
                            if response_msg is None:
                                # Create first message
                                response_msg = await message.channel.send(
                                    formatted_response,
                                    reference=message,
                                    allowed_mentions=discord.AllowedMentions.none()
                                )
                                message_chunks = [response_msg]
                            else:
                                # Update existing message
                                await response_msg.edit(content=formatted_response)
                        else:
                            # Response is too long - need to split into multiple messages
                            chunks = split_message_smart(formatted_response, 2000)
                            
                            # Update existing messages and create new ones as needed
                            for i, chunk_content in enumerate(chunks):
                                if i < len(message_chunks):
                                    # Update existing message
                                    await message_chunks[i].edit(content=chunk_content)
                                else:
                                    # Create new message
                                    new_msg = await message.channel.send(
                                        chunk_content,
                                        reference=message if i == 0 and len(message_chunks) == 0 else None,
                                        allowed_mentions=discord.AllowedMentions.none()
                                    )
                                    message_chunks.append(new_msg)
                                    await asyncio.sleep(0.3)  # Small delay between messages
                            
                            # If we have fewer chunks now, delete extra messages
                            while len(message_chunks) > len(chunks):
                                extra_msg = message_chunks.pop()
                                try:
                                    await extra_msg.delete()
                                except:
                                    pass
                        
                        last_update = current_time
                    except discord.errors.HTTPException as e:
                        if "Must be 2000 or fewer in length" in str(e):
                            # Force split the message
                            formatted_response = convert_latex_to_discord(collected_response)
                            chunks = split_message_smart(formatted_response, 2000)
                            
                            if response_msg is None:
                                # Send as multiple messages from start
                                for i, chunk_content in enumerate(chunks):
                                    msg = await message.channel.send(
                                        chunk_content,
                                        reference=message if i == 0 else None,
                                        allowed_mentions=discord.AllowedMentions.none()
                                    )
                                    message_chunks.append(msg)
                                    if i > 0:
                                        await asyncio.sleep(0.3)
                                if message_chunks:
                                    response_msg = message_chunks[0]
                            last_update = current_time
                        else:
                            logger.error(f"Failed to update stream message: {e}")
                            
            # Final update and process completion
            if collected_response:
                final_response = convert_latex_to_discord(collected_response)
                
                # Final update with complete response
                if len(final_response) <= 2000:
                    if response_msg is None:
                        response_msg = await message.channel.send(
                            final_response,
                            reference=message,
                            allowed_mentions=discord.AllowedMentions.none()
                        )
                    else:
                        await response_msg.edit(content=final_response)
                        # Clean up any extra messages
                        for i in range(1, len(message_chunks)):
                            try:
                                await message_chunks[i].delete()
                            except:
                                pass
                else:
                    # Send as multiple messages using the smart splitting
                    await send_long_message_with_reference(
                        message.channel, 
                        final_response, 
                        message
                    )
                    # Clean up streaming messages if any
                    for msg in message_chunks:
                        try:
                            await msg.delete()
                        except:
                            pass
                
                # Save to memory store
                if _memory_store:
                    _memory_store.add_message(user_id, {"role": "user", "content": combined_text})
                    _memory_store.add_message(user_id, {"role": "assistant", "content": collected_response})
                    
                # Deduct credits after successful completion
                if _use_mongodb_auth and model_info:
                    await deduct_credits(user_id, model_info.get("credit_cost", 0))
                    
        else:
            # NON-STREAMING RESPONSE - Use enhanced long message handler with image support
            ok, resp = await asyncio.get_event_loop().run_in_executor(
                None,
                _call_api.call_openai_proxy,
                payload_messages,
                user_model
            )
            
            if ok and resp:
                # Use the smart long message handler
                await send_long_message_with_reference(message.channel, resp, message)
                
                # Save to memory store
                if _memory_store:
                    _memory_store.add_message(user_id, {"role": "user", "content": combined_text})
                    _memory_store.add_message(user_id, {"role": "assistant", "content": resp})
                
                # Deduct credits after successful completion    
                if _use_mongodb_auth and model_info:
                    await deduct_credits(user_id, model_info.get("credit_cost", 0))
            else:
                error_msg = resp or "Unknown error"
                await message.channel.send(
                    f"⚠️ Error: {error_msg}",
                    reference=message,
                    allowed_mentions=discord.AllowedMentions.none()
                )

    except Exception as e:
        logger.exception(f"Error in request processing for user {user_id}")
        await message.channel.send(
            f"⚠️ Internal error: {e}",
            reference=message,
            allowed_mentions=discord.AllowedMentions.none()
        )

# ------------------------------------------------------------------
# SLASH COMMAND HANDLERS
# ------------------------------------------------------------------

@app_commands.command(name="help", description="Show available commands")
async def help_slash(interaction: discord.Interaction):
    """Show help information"""
    is_owner = False
    try:
        is_owner = await _bot.is_owner(interaction.user)
    except Exception:
        pass

    lines = [
        "**Available commands:**",
        "`/getid <>` – Show your ID (or a mention). (everyone)",
        "`/ping` – Check bot responsiveness. (everyone)",
        "",
        "**Configuration commands (authorized users):**",
        "`/set model <model>` – Set your preferred AI model.",
        "`/set sys_prompt <prompt>` – Set your system prompt.", 
        "`/show profile <user>` – Show user configuration.",
        "`/show sys_prompt <user>` - View system prompt.",
        "`/show models` – Show all supported models.",
        "`/clearmemory <user>` – Clear conversation history.",
        "",
        "**🖼️ Image Analysis (Gemini models only):**",
        "• Attach images (JPG, PNG, GIF, WebP, BMP) with your message",
        "• Images work with regular Gemini models (not Live models)",
        "• Max 10MB per image, multiple images supported"
    ]

    if is_owner:
        lines += [
            "",
            "**Owner‑only commands:**",
            "`/auth <user>` – Add a user to authorized list.",
            "`/deauth <user>` – Remove user from authorized list.",
            "`/show auth` – List authorized users.",
            "`/memory <user>` – View conversation history.",
            "",
            "**Model management (owner only):**",
            "`/add model <name> <cost> <level>` – Add a new model",
            "  - cost: Cost in credits per use",
            "  - level: Required user level (0=Basic, 1=Advanced, 2=Premium, 3=Ultimate)",
            "`/remove model <name>` – Remove a model",
            "`/edit model <name> <cost> <level>` – Edit model settings",
            "",
            "**Profile Model Management (owner only):**",
            "`/add pmodel <name>` - Create new profile model",
            "`/edit pmodel <name> <field> <value>` - Edit profile model",
            "`/show pmodels` - List all profile models"
        ]

    await interaction.response.send_message("\n".join(lines))

@app_commands.command(name="getid", description="Get user ID")
@app_commands.describe(member="The member to get ID for (optional)")
async def getid_slash(interaction: discord.Interaction, member: discord.Member = None):
    """Get user ID"""
    if member is None:
        await interaction.response.send_message(f"Your ID: {interaction.user.id}")
    else:
        await interaction.response.send_message(f"{member} ID: {member.id}")

@app_commands.command(name="ping", description="Check bot responsiveness")
async def ping_slash(interaction: discord.Interaction):
    """Ping command"""
    start_time = time.perf_counter()
    await interaction.response.send_message("Pinging...")
    end_time = time.perf_counter()
    
    # Calculate response time (ms)
    latency_ms = round((end_time - start_time) * 1000)
    
    # Get bot's WebSocket latency (if available)
    ws_latency = round(_bot.latency * 1000) if _bot.latency else "N/A"
    
    # Update message with detailed info
    content = f"Pong! \nResponse: {latency_ms} ms\nWebSocket: {ws_latency} ms"
    await interaction.edit_original_response(content=content)

# ------------------------------------------------------------------
# Set commands group
# ------------------------------------------------------------------
set_group = app_commands.Group(name="set", description="Configure bot settings")

@set_group.command(name="model", description="Set your preferred AI model")
@app_commands.describe(model="The model to use")
async def set_model_slash(interaction: discord.Interaction, model: str):
    """Set user model"""
    if not await is_authorized_user(interaction.user):
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)
        return
    
    # Check if value is a profile model first
    profile_model = None
    if _use_mongodb_auth:
        profile_model = _mongodb_store.get_profile_model(model.strip())

    if profile_model:
        success, message = _user_config_manager.set_user_model(interaction.user.id, model.strip())
        await interaction.response.send_message(f"Successfully set model to profile: {model.strip()}")
        return
    else:
        # Regular model handling
        available, error = _call_api.is_model_available(model.strip())
        if not available:
            await interaction.response.send_message(error, ephemeral=True)
            return

        success, message = _user_config_manager.set_user_model(interaction.user.id, model.strip())
        await interaction.response.send_message(message)

@set_group.command(name="sys_prompt", description="Set your system prompt")
@app_commands.describe(prompt="The system prompt to use")
async def set_sys_prompt_slash(interaction: discord.Interaction, prompt: str):
    """Set user system prompt"""
    if not await is_authorized_user(interaction.user):
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)
        return
    
    success, message = _user_config_manager.set_user_system_prompt(interaction.user.id, prompt)
    await interaction.response.send_message(message)

@set_group.command(name="level", description="Set user access level (owner only)")
@app_commands.describe(
    user="The user to set level for",
    level="Access level (0=Basic, 1=Advanced, 2=Premium, 3=Ultimate)"
)
@app_commands.choices(level=[
    app_commands.Choice(name="Basic (0)", value=0),
    app_commands.Choice(name="Advanced (1)", value=1),
    app_commands.Choice(name="Premium (2)", value=2),
    app_commands.Choice(name="Ultimate (3)", value=3)
])
async def set_level_slash(interaction: discord.Interaction, user: discord.Member, level: int):
    """Set user level (owner only)"""
    try:
        is_owner = await _bot.is_owner(interaction.user)
    except Exception:
        is_owner = False
        
    if not is_owner:
        await interaction.response.send_message("Only the bot owner can set user levels.", ephemeral=True)
        return
        
    if _use_mongodb_auth:
        success = _mongodb_store.set_user_level(user.id, level)
        if success:
            level_names = {0: "Basic", 1: "Advanced", 2: "Premium", 3: "Ultimate"}
            await interaction.response.send_message(f"Set {user}'s level to {level_names[level]} (Level {level})")
        else:
            await interaction.response.send_message("Failed to set user level", ephemeral=True)
    else:
        await interaction.response.send_message("User levels require MongoDB mode", ephemeral=True)

# ------------------------------------------------------------------
# Show commands group
# ------------------------------------------------------------------
show_group = app_commands.Group(name="show", description="Display information")

@show_group.command(name="profile", description="Show user profile")
@app_commands.describe(user="The user to show profile for (optional)")
async def show_profile_slash(interaction: discord.Interaction, user: discord.Member = None):
    """Show user profile"""
    target_user = user or interaction.user

    if target_user != interaction.user:
        try:
            is_owner = await _bot.is_owner(interaction.user)
        except Exception:
            is_owner = False

        if not is_owner:
            await interaction.response.send_message("You can only view your own profile.", ephemeral=True)
            return

    # Gather config for the target user
    user_config = _user_config_manager.get_user_config(target_user.id)
    
    # Get additional info
    model = user_config["model"]
    credit = user_config.get("credit", 0)
    access_level = user_config.get("access_level", 0)

    # Level description
    level_desc = {
        0: "Basic (Level 0)",
        1: "Advanced (Level 1)", 
        2: "Premium (Level 2)",
        3: "Ultimate (Level 3)"
    }.get(access_level, f"Unknown (Level {access_level})")

    # Check if model supports images
    model_features = []
    if is_gemini_model(model) and not is_gemini_live_model(model):
        model_features.append("🖼️ Image Analysis")
    if is_gemini_live_model(model) or "live-preview" in model:
        model_features.append("⚡ Live Streaming")
    
    features_text = f"\n**Features**: {', '.join(model_features)}" if model_features else ""

    # Build profile display without system prompt
    lines = [
        f"**Profile for {target_user}:**",
        f"**Current Model**: {model}{features_text}",
        f"**Credit Balance**: {credit}",
        f"**Access Level**: {level_desc}",
        "",
        "Use `/show sys_prompt` to view system prompt."
    ]

    await interaction.response.send_message("\n".join(lines))

@show_group.command(name="sys_prompt", description="Show system prompt")
@app_commands.describe(user="The user to show system prompt for (optional)")
async def show_sys_prompt_slash(interaction: discord.Interaction, user: discord.Member = None):
    """Show user system prompt"""
    target_user = user or interaction.user
    
    # Check if viewer is owner when viewing other's prompt
    if target_user != interaction.user:
        try:
            is_owner = await _bot.is_owner(interaction.user)
        except Exception:
            is_owner = False

        if not is_owner:
            await interaction.response.send_message("Only the bot owner can view other users' system prompts.", ephemeral=True)
            return
            
    # Get system prompt
    user_config = _user_config_manager.get_user_config(target_user.id)
    prompt = user_config["system_prompt"]
    
    # Format display
    lines = [   
        f"**System Prompt for {target_user}:**",
        "```",
        prompt,
        "```",
        "",
        "This is the prompt used to guide the AI's responses."
    ]
    
    await interaction.response.send_message("\n".join(lines))

@show_group.command(name="models", description="Show available models")
async def show_models_slash(interaction: discord.Interaction):
    """Show available models"""
    if _use_mongodb_auth:
        # Get models with their details from MongoDB
        all_models = _mongodb_store.list_all_models()
        all_pmodels = _mongodb_store.list_profile_models() 

        # Combine regular models and profile models
        if all_models or all_pmodels:
            models_info = []
            
            # Combine and sort all models
            combined_models = []
            
            # Add regular models
            for model in all_models:
                model_name = model.get("model_name", "Unknown")
                features = []
                if is_gemini_model(model_name):
                    if is_gemini_live_model(model_name):
                        features.append("⚡Live")
                    else:
                        features.append("🖼️IMG")
                
                combined_models.append({
                    "name": model_name,
                    "credit_cost": model.get("credit_cost", 0),
                    "access_level": model.get("access_level", 0),
                    "features": features
                })
            
            # Add profile models
            for pmodel in all_pmodels:
                pmodel_name = pmodel.get("name", "Unknown")
                base_model = pmodel.get("base_model", "")
                features = []
                if is_gemini_model(base_model):
                    if is_gemini_live_model(base_model):
                        features.append("⚡Live")
                    else:
                        features.append("🖼️IMG")
                        
                combined_models.append({
                    "name": pmodel_name,
                    "credit_cost": pmodel.get("credit_cost", 0),
                    "access_level": pmodel.get("access_level", 0),
                    "features": features
                })
            
            # Sort all models together
            sorted_models = sorted(
                combined_models, 
                key=lambda x: (-x["access_level"], -x["credit_cost"], x["name"])
            )
            
            current_level = None
            for model in sorted_models:
                model_name = model["name"]
                credit_cost = model["credit_cost"]
                access_level = model["access_level"]
                features = model["features"]
                level_names = {0: "Basic", 1: "Advanced", 2: "Premium", 3: "Ultimate"}
                level_name = level_names.get(access_level, f"Level {access_level}")
                
                # Add level header if level changed
                if current_level != access_level:
                    models_info.append(f"\n**{level_name} Models:**")
                    current_level = access_level
                
                feature_text = f" {' '.join(features)}" if features else ""
                models_info.append(f"• `{model_name}` - {credit_cost} credits{feature_text}")

            lines = [
                "**Available AI Models:**",
                *models_info,
                "",
                "**Legend:** 🖼️IMG = Image support, ⚡Live = Live streaming",
                "",
                "Use `/set model <model_name>` to change your model."
            ]
        else:
            lines = ["No models found."]

    else:
        # Fallback for file mode
        supported_models = _user_config_manager.get_supported_models()
        models_list = []
        for model in sorted(supported_models):
            features = []
            if is_gemini_model(model):
                if is_gemini_live_model(model):
                    features.append("⚡Live")
                else:
                    features.append("🖼️IMG")
            
            feature_text = f" {' '.join(features)}" if features else ""
            models_list.append(f"• `{model}`{feature_text}")
            
        lines = [
            "**Supported AI Models:**",
            "\n".join(models_list),
            "",
            "**Legend:** 🖼️IMG = Image support, ⚡Live = Live streaming",
            "",
            "Use `/set model <model_name>` to change your model."
        ]

    await interaction.response.send_message("\n".join(lines))

@show_group.command(name="auth", description="List authorized users (owner only)")
async def show_auth_slash(interaction: discord.Interaction):
    """Show authorized users (owner only)"""
    try:
        is_owner = await _bot.is_owner(interaction.user)
    except Exception:
        is_owner = False

    if not is_owner:
        await interaction.response.send_message("Only the bot owner can view the authorized users list.", ephemeral=True)
        return

    if not _authorized_users:
        await interaction.response.send_message("Authorized users list is empty.")
        return

    body = "\n".join(str(x) for x in sorted(_authorized_users))
    if len(body) > 1900:
        # Create file if too long
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Authorized Users:\n")
            f.write(body)
            temp_path = f.name
        try:
            await interaction.response.send_message(
                "Too long data, sending authorized_users.txt file.",
                file=discord.File(temp_path, filename="authorized_users.txt")
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)
    else:
        await interaction.response.send_message(f"**Authorized users list:**\n{body}")

@show_group.command(name="pmodels", description="List profile models (owner only)")
async def show_pmodels_slash(interaction: discord.Interaction):
    """Show profile models (owner only)"""
    try:
        is_owner = await _bot.is_owner(interaction.user)
    except Exception:
        is_owner = False

    if not is_owner:
        await interaction.response.send_message("Only the bot owner can view profile models.", ephemeral=True)
        return

    all_profiles = _mongodb_store.list_profile_models()
    if all_profiles:
        lines = ["**Profile Models:**"]
        for p in all_profiles:
            name = p.get("name", "Unknown")
            base = p.get("base_model", "Unknown")
            cost = p.get("credit_cost", 0)
            level = p.get("access_level", 0)
            is_live = "✅" if p.get("is_live") else "❌"
            
            features = []
            if is_gemini_model(base):
                if is_gemini_live_model(base):
                    features.append("⚡")
                else:
                    features.append("🖼️")
            feature_text = f" {' '.join(features)}" if features else ""
            
            lines.append(f"• `{name}` -> {base} (Cost: {cost}, Level: {level}, Live: {is_live}){feature_text}")
    else:
        lines = ["No profile models found."]
        
    await interaction.response.send_message("\n".join(lines))

# ------------------------------------------------------------------
# Owner-only commands
# ------------------------------------------------------------------

@app_commands.command(name="auth", description="Add user to authorized list (owner only)")
@app_commands.describe(user="The user to authorize")
async def auth_slash(interaction: discord.Interaction, user: discord.Member):
    """Add user to authorized list"""
    try:
        is_owner = await _bot.is_owner(interaction.user)
    except Exception:
        is_owner = False
        
    if not is_owner:
        await interaction.response.send_message("Only the bot owner can authorize users.", ephemeral=True)
        return

    uid = user.id
    if uid in _authorized_users:
        await interaction.response.send_message(f"User {user} is already authorized.")
        return

    success = add_authorized_user(uid)
    if success:
        await interaction.response.send_message(f"Added {user} to authorized list.")
    else:
        await interaction.response.send_message(f"Failed to add {user} to authorized list.")

@app_commands.command(name="deauth", description="Remove user from authorized list (owner only)")
@app_commands.describe(user="The user to deauthorize")
async def deauth_slash(interaction: discord.Interaction, user: discord.Member):
    """Remove user from authorized list"""
    try:
        is_owner = await _bot.is_owner(interaction.user)
    except Exception:
        is_owner = False
        
    if not is_owner:
        await interaction.response.send_message("Only the bot owner can deauthorize users.", ephemeral=True)
        return

    uid = user.id
    if uid not in _authorized_users:
        await interaction.response.send_message(f"User {user} is not in the authorized list.")
        return

    success = remove_authorized_user(uid)
    if success:
        await interaction.response.send_message(f"Removed {user} from authorized list.")
    else:
        await interaction.response.send_message(f"Failed to remove {user} from authorized list.")

@app_commands.command(name="memory", description="View conversation history (owner only)")
@app_commands.describe(user="The user to view memory for (optional)")
async def memory_slash(interaction: discord.Interaction, user: discord.Member = None):
    """View the conversation history of user (or the author)."""
    try:
        is_owner = await _bot.is_owner(interaction.user)
    except Exception:
        is_owner = False
        
    if not is_owner:
        await interaction.response.send_message("Only the bot owner can view memory.", ephemeral=True)
        return

    target = user or interaction.user
    if _memory_store is None:
        await interaction.response.send_message("Memory feature not initialized.", ephemeral=True)
        return

    mem = _memory_store.get_user_messages(target.id)
    if not mem:
        await interaction.response.send_message(f"No memory for {target}.")
        return

    lines = []
    for i, msg in enumerate(mem[-10:], start=1):
        content = msg["content"]
        preview = (content[:120] + "…") if len(content) > 120 else content
        lines.append(f"{i:02d}. **{msg['role']}**: {preview}")

    await interaction.response.send_message("\n".join(lines))

@app_commands.command(name="clearmemory", description="Clear conversation history")
@app_commands.describe(user="The user to clear memory for (optional - owner only for others)")
async def clearmemory_slash(interaction: discord.Interaction, user: discord.Member = None):
    """Clear conversation history - users can clear their own, owners can clear anyone's."""
    target_user = user or interaction.user
    
    # Check if trying to clear someone else's memory
    if target_user != interaction.user:
        try:
            is_owner = await _bot.is_owner(interaction.user)
        except Exception:
            is_owner = False
            
        if not is_owner:
            await interaction.response.send_message("You can only clear your own memory. Only the bot owner can clear other users' memory.", ephemeral=True)
            return
    
    # Check if user is authorized (for clearing their own memory)
    if target_user == interaction.user and not await is_authorized_user(interaction.user):
        await interaction.response.send_message("You do not have permission to use this command.", ephemeral=True)
        return

    if _memory_store is None:
        await interaction.response.send_message("Memory feature not initialized.", ephemeral=True)
        return

    _memory_store.clear_user(target_user.id)
    
    if target_user == interaction.user:
        await interaction.response.send_message("Cleared your conversation memory.")
    else:
        await interaction.response.send_message(f"Cleared memory for {target_user}.")

# ------------------------------------------------------------------
# Add/Remove/Edit commands groups (owner only)
# ------------------------------------------------------------------
add_group = app_commands.Group(name="add", description="Add resources (owner only)")

@add_group.command(name="model", description="Add a new model (owner only)")
@app_commands.describe(
    model_name="The model name",
    credit_cost="Cost in credits per use",
    access_level="Required user level (0=Basic, 1=Advanced, 2=Premium, 3=Ultimate)"
)
@app_commands.choices(access_level=[
    app_commands.Choice(name="Basic (0)", value=0),
    app_commands.Choice(name="Advanced (1)", value=1),
    app_commands.Choice(name="Premium (2)", value=2),
    app_commands.Choice(name="Ultimate (3)", value=3)
])
async def add_model_slash(interaction: discord.Interaction, model_name: str, credit_cost: int, access_level: int):
    """Add a new model (owner only)"""
    try:
        is_owner = await _bot.is_owner(interaction.user)
    except Exception:
        is_owner = False
        
    if not is_owner:
        await interaction.response.send_message("Only the bot owner can add models.", ephemeral=True)
        return

    if credit_cost < 0:
        await interaction.response.send_message("Credit cost must be positive.", ephemeral=True)
        return
        
    # Check if MongoDB is enabled
    if not _config.USE_MONGODB:
        await interaction.response.send_message("Model management requires MongoDB mode to be enabled.", ephemeral=True)
        return
    
    success, message = _mongodb_store.add_supported_model(model_name, credit_cost, access_level)
    await interaction.response.send_message(message)

@add_group.command(name="credit", description="Add credits to user (owner only)")
@app_commands.describe(
    user="The user to add credits to",
    amount="Amount of credits to add"
)
async def add_credit_slash(interaction: discord.Interaction, user: discord.Member, amount: int):
    """Add credits to user (owner only)"""
    try:
        is_owner = await _bot.is_owner(interaction.user)
    except Exception:
        is_owner = False
        
    if not is_owner:
        await interaction.response.send_message("Only the bot owner can add credits.", ephemeral=True)
        return

    if amount <= 0:
        await interaction.response.send_message("Amount must be positive.", ephemeral=True)
        return
            
    if _use_mongodb_auth:
        success, new_balance = _mongodb_store.add_user_credit(user.id, amount)
        if success:
            await interaction.response.send_message(f"Added {amount} credits to {user}'s balance. New balance: {new_balance}")
        else:
            await interaction.response.send_message("Failed to add credits", ephemeral=True)
    else:
        await interaction.response.send_message("Credit system requires MongoDB mode", ephemeral=True)

@add_group.command(name="pmodel", description="Add a profile model (owner only)")
@app_commands.describe(
    name="The profile model name",
    base_model="The base model to use",
    sys_prompt="The system prompt",
    credit_cost="Cost in credits (default: 0)",
    access_level="Required access level (default: 0)",
    is_live="Enable live preview (default: False)"
)
@app_commands.choices(
    access_level=[
        app_commands.Choice(name="Basic (0)", value=0),
        app_commands.Choice(name="Advanced (1)", value=1),
        app_commands.Choice(name="Premium (2)", value=2),
        app_commands.Choice(name="Ultimate (3)", value=3)
    ]
)
async def add_pmodel_slash(interaction: discord.Interaction, name: str, base_model: str, sys_prompt: str, 
                         credit_cost: int = 0, access_level: int = 0, is_live: bool = False):
    """Add a profile model (owner only)"""
    try:
        is_owner = await _bot.is_owner(interaction.user)
    except Exception:
        is_owner = False
        
    if not is_owner:
        await interaction.response.send_message("Only the bot owner can add profile models.", ephemeral=True)
        return

    if not _config.USE_MONGODB:
        await interaction.response.send_message("Profile models require MongoDB mode.", ephemeral=True)
        return

    success, message = _mongodb_store.add_profile_model(
        name=name,
        base_model=base_model,
        sys_prompt=sys_prompt,
        credit_cost=credit_cost,
        access_level=access_level,
        is_live=is_live
    )
    
    await interaction.response.send_message(message)

# Remove commands group
remove_group = app_commands.Group(name="remove", description="Remove resources (owner only)")

@remove_group.command(name="model", description="Remove a model (owner only)")
@app_commands.describe(model_name="The model name to remove")
async def remove_model_slash(interaction: discord.Interaction, model_name: str):
    """Remove a model (owner only)"""
    try:
        is_owner = await _bot.is_owner(interaction.user)
    except Exception:
        is_owner = False
        
    if not is_owner:
        await interaction.response.send_message("Only the bot owner can remove models.", ephemeral=True)
        return
    
    # Check if MongoDB is enabled
    if not _config.USE_MONGODB:
        await interaction.response.send_message("Model management requires MongoDB mode to be enabled.", ephemeral=True)
        return
    
    success, message = _mongodb_store.remove_supported_model(model_name)
    await interaction.response.send_message(message)

# Edit commands group
edit_group = app_commands.Group(name="edit", description="Edit resources (owner only)")

@edit_group.command(name="model", description="Edit a model (owner only)")
@app_commands.describe(
    model_name="The model name to edit",
    credit_cost="New cost in credits per use",
    access_level="New required user level"
)
@app_commands.choices(access_level=[
    app_commands.Choice(name="Basic (0)", value=0),
    app_commands.Choice(name="Advanced (1)", value=1),
    app_commands.Choice(name="Premium (2)", value=2),
    app_commands.Choice(name="Ultimate (3)", value=3)
])
async def edit_model_slash(interaction: discord.Interaction, model_name: str, credit_cost: int, access_level: int):
    """Edit a model (owner only)"""
    try:
        is_owner = await _bot.is_owner(interaction.user)
    except Exception:
        is_owner = False
        
    if not is_owner:
        await interaction.response.send_message("Only the bot owner can edit models.", ephemeral=True)
        return

    if credit_cost < 0:
        await interaction.response.send_message("Credit cost must be positive.", ephemeral=True)
        return
        
    # Check if MongoDB is enabled
    if not _config.USE_MONGODB:
        await interaction.response.send_message("Model management requires MongoDB mode to be enabled.", ephemeral=True)
        return
    
    success, message = _mongodb_store.edit_supported_model(model_name, credit_cost, access_level)
    await interaction.response.send_message(message)

@edit_group.command(name="pmodel", description="Edit a profile model (owner only)")
@app_commands.describe(
    name="The profile model name to edit",
    field="The field to edit",
    value="The new value"
)
@app_commands.choices(field=[
    app_commands.Choice(name="base_model", value="base_model"),
    app_commands.Choice(name="sys_prompt", value="sys_prompt"),
    app_commands.Choice(name="credit_cost", value="credit_cost"),
    app_commands.Choice(name="access_level", value="access_level"),
    app_commands.Choice(name="is_live", value="is_live")
])
async def edit_pmodel_slash(interaction: discord.Interaction, name: str, field: str, value: str):
    """Edit a profile model (owner only)"""
    try:
        is_owner = await _bot.is_owner(interaction.user)
    except Exception:
        is_owner = False
        
    if not is_owner:
        await interaction.response.send_message("Only the bot owner can edit profile models.", ephemeral=True)
        return

    if not _config.USE_MONGODB:
        await interaction.response.send_message("Profile models require MongoDB mode.", ephemeral=True)
        return

    success, message = _mongodb_store.edit_profile_model(name, field, value)
    await interaction.response.send_message(message)

# ------------------------------------------------------------------
# on_message listener – central dispatch point
# ------------------------------------------------------------------
async def on_message(message: discord.Message):
    try:
        logger.info(
            "on_message invoked: func id=%s module=%s qualname=%s author=%s content=%s",
            hex(id(on_message)),
            on_message.__module__,
            getattr(on_message, "__qualname__", "?"),
            f"{message.author}({getattr(message.author, 'id', None)})",
            (message.content or "")[:120],
        )
    except Exception:
        pass

    if message.author.bot:
        return

    content = (message.content or "").strip()

    # Skip prefix commands completely - slash commands handle everything now
    if content.startswith(";"):
        return

    # Default trigger (DM or mention) - for AI responses
    authorized = await is_authorized_user(message.author)
    attachments = list(message.attachments or [])

    if not should_respond_default(message):
        return

    if not authorized:
        try:
            await message.channel.send("You do not have permission to use this bot.", 
                                     allowed_mentions=discord.AllowedMentions.none())
        except Exception:
            logger.exception("Failed to send unauthorized message")
        return

    # ------------------------------------------------------------------
    # Build the user prompt (after stripping the bot mention)
    # ------------------------------------------------------------------
    user_text = content
    if _bot.user in message.mentions:
        user_text = re.sub(rf"<@!?{_bot.user.id}>", "", content).strip()

    # ------------------------------------------------------------------
    # Handle attachments with enhanced image support
    # ------------------------------------------------------------------
    attachment_text = ""
    if attachments:
        attachment_data = await _read_attachments_enhanced(attachments)
        attachment_text = attachment_data["text_summary"]
        
        # Log image processing
        if attachment_data["has_images"]:
            image_count = len([img for img in attachment_data["images"] if not img.get("skipped")])
            logger.info(f"User {message.author.id} sent {image_count} valid images")

    final_user_text = (attachment_text + user_text).strip()
    if not final_user_text and not any(not img.get("skipped") for img in attachment_data.get("images", [])):
        await message.channel.send(
            "Please send a message (mention me or DM me) with your question or attach some files/images.",
            allowed_mentions=discord.AllowedMentions.none(),
        )
        return

    # ------------------------------------------------------------------
    # Add request to queue instead of processing directly
    # ------------------------------------------------------------------
    try:
        success, status_message = await _request_queue.add_request(message, final_user_text)
        if not success:
            await message.channel.send(status_message, allowed_mentions=discord.AllowedMentions.none())
            return
        
        # Send status message if not immediately processing
        queue_size = _request_queue._queue.qsize()
        processing_count = len(_request_queue._processing_users)
        is_owner = await _request_queue.is_owner(message.author)
        
        if queue_size > 1 or processing_count > 0:
            await message.channel.send(
                status_message,
                reference=message,
                allowed_mentions=discord.AllowedMentions.none()
            )
    
    except Exception as e:
        logger.exception("Error adding request to queue")
        await message.channel.send(
            f"Error adding request to queue: {e}",
            allowed_mentions=discord.AllowedMentions.none()
        )

# ------------------------------------------------------------------
# Setup – register commands, listeners, load data (updated for slash commands)
# ------------------------------------------------------------------
def setup(bot: commands.Bot, call_api_module, config_module):
    global _bot, _call_api, _config, _authorized_users, _memory_store, _user_config_manager, _request_queue
    global _use_mongodb_auth, _mongodb_store

    _bot = bot
    _call_api = call_api_module
    _config = config_module
    
    # Initialize storage backend
    _config.init_storage()
    
    # Check if we're using MongoDB
    _use_mongodb_auth = _config.USE_MONGODB
    if _use_mongodb_auth:
        from mongodb_store import get_mongodb_store
        _mongodb_store = get_mongodb_store()
        logger.info("Using MongoDB for data storage")
    else:
        _mongodb_store = None
        logger.info("Using file-based storage (legacy mode)")
    
    # Initialize managers
    _user_config_manager = get_user_config_manager()
    _request_queue = get_request_queue()

    # Setup queue
    _request_queue.set_bot(bot)
    _request_queue.set_process_callback(process_ai_request)

    # Load authorized users
    _authorized_users = load_authorized_users()
    logger.info("Functions module initialized. Authorized users: %s", sorted(_authorized_users))

    # Initialize memory store
    _memory_store = MemoryStore()
    if not _use_mongodb_auth:
        logger.info("Memory store: %d users cached", len(_memory_store._cache))
    else:
        logger.info("Memory store initialized with MongoDB backend")

    # ------------------------------------------------------------------
    # Clear existing slash commands and add new ones
    # ------------------------------------------------------------------
    bot.tree.clear_commands(guild=None)
    
    # Add individual slash commands
    bot.tree.add_command(help_slash)
    bot.tree.add_command(getid_slash)
    bot.tree.add_command(ping_slash)
    
    # Add command groups
    bot.tree.add_command(set_group)
    bot.tree.add_command(show_group)
    bot.tree.add_command(add_group)
    bot.tree.add_command(remove_group)
    bot.tree.add_command(edit_group)
    
    # Add owner-only commands
    bot.tree.add_command(auth_slash)
    bot.tree.add_command(deauth_slash)
    bot.tree.add_command(memory_slash)
    bot.tree.add_command(clearmemory_slash)

    # ------------------------------------------------------------------
    # Register on_message listener if not already present
    # ------------------------------------------------------------------
    already = False
    try:
        existing = list(getattr(bot, "_listeners", {}).get("on_message", []))
        for l in existing:
            if getattr(l, "__qualname__", None) == on_message.__qualname__ \
               and getattr(l, "__module__", None) == on_message.__module__:
                already = True
                break
    except Exception:
        pass

    if not already:
        bot.add_listener(on_message, "on_message")
        logger.info("on_message listener registered.")
    else:
        logger.info("on_message listener already registered; not adding again.")

    logger.info("Slash commands registered successfully with image support")