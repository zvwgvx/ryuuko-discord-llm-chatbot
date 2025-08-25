#!/usr/bin/env python3
# coding: utf-8
# ────────────────────────────────────────────────────────────────────────
# Bot helper / command registry
# Uses MemoryStore for per‑user conversation history
# Uses UserConfigManager for per-user model and system prompt settings
# Uses MongoDB for model management
# ────────────────────────────────────────────────────────────────────────

import re
import json
import logging
import asyncio
import time
from pathlib import Path
from typing import Set, Optional, List, Dict
from datetime import datetime, timezone, timedelta

import discord
from discord.ext import commands

# ───────────────────────────────────────────────────────
# ***Absolute import – no package, so we use the plain module name ***
from memory_store import MemoryStore
from user_config import get_user_config_manager
from request_queue import get_request_queue

logger = logging.getLogger("discord-openai-proxy.functions")

# ---------------------------- module‑level state -----------------------------
_bot: Optional[commands.Bot] = None
_call_api = None
_config = None
_user_config_manager = None
_request_queue = None

# ---------------------------------------------------------------
# Persistence helpers – authorized user IDs
# ---------------------------------------------------------------
_authorized_users: Set[int] = set()

# MongoDB storage globals
_use_mongodb_auth = False
_mongodb_store = None

# ---------------------------------------------------------------
# Attachment handling constants
# ---------------------------------------------------------------
FILE_MAX_BYTES = 200 * 1024          # 200 KB per file
MAX_CHARS_PER_FILE = 10_000
ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".java", ".c", ".cpp", ".h",
    ".json", ".yaml", ".yml", ".csv", ".rs", ".go", ".rb",
    ".sh", ".html", ".css", ".ts", ".ini", ".toml",
}

# ---------------------------------------------------------------
# Optional memory store
# ---------------------------------------------------------------
_memory_store: Optional[MemoryStore] = None

# ------------------------------------------------------------------
# Persistence helpers – authorized users
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

# ------------------------------------------------------------------
# Attachment helpers
# ------------------------------------------------------------------
async def _read_attachments_as_text(attachments: List[discord.Attachment]) -> List[Dict]:
    """Return a list of dicts describing each attachment that looks like text."""
    result = []
    for att in attachments:
        entry = {"filename": att.filename, "text": "", "skipped": False, "reason": None}

        # quick size check
        try:
            size = int(getattr(att, "size", 0) or 0)
        except Exception:
            size = 0

        ext = (Path(att.filename).suffix or "").lower()
        content_type = getattr(att, "content_type", "") or ""

        # filter by content‑type / extension
        if not (
            content_type.startswith("text")
            or content_type in ("application/json", "application/javascript")
            or ext in ALLOWED_EXTENSIONS
        ):
            entry["skipped"] = True
            entry["reason"] = f"unsupported file type ({content_type!r}, {ext!r})"
            result.append(entry)
            continue

        if size and size > FILE_MAX_BYTES:
            entry["skipped"] = True
            entry["reason"] = f"file too large ({size} bytes)"
            result.append(entry)
            continue

        try:
            b = await att.read()
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
            result.append(entry)
        except Exception as e:
            logger.exception("Error reading attachment %s", att.filename)
            entry["skipped"] = True
            entry["reason"] = f"read error: {e}"
            result.append(entry)

    return result

# ------------------------------------------------------------------
# Command handlers
# ------------------------------------------------------------------
async def help_cmd(ctx: commands.Context):
    is_owner = False
    try:
        is_owner = await _bot.is_owner(ctx.author)
    except Exception:
        pass

    lines = [
        "**Available commands:**",
        "`;getid [@member]` – Show your ID (or a mention). (everyone)",
        "`;ping` – Check bot responsiveness. (everyone)",
        "",
        "**Configuration commands (authorized users):**",
        "`;set model <model>` – Set your preferred AI model.",
        "`;set sys_prompt <prompt>` – Set your system prompt.", 
        "`;show profile` – Show your current configuration.",
        "`;show sys_prompt [@user]` - View system prompt.",
        "`;show model` – Show all supported models."
    ]

    if is_owner:
        lines += [
            "",
            "**Owner‑only commands:**",
            "`;auth <id|@mention>` – Add a user to authorized list.",
            "`;deauth <id|@mention>` – Remove user from authorized list.",
            "`;show auth` – List authorized users.",
            "`;memory [@user]` – View conversation history.",
            "`;clearmemory [@user]` – Clear conversation history.",
            "",
            "**Model management (owner only):**",
            "`;add model <model_name> <credit_cost> <access_level>` – Add a new model",
            "  - credit_cost: Cost in credits per use",
            "  - access_level: Required user level (0=Basic, 1=Advanced, 2=Premium, 3=Ultimate)",
            "`;remove model <model_name>` – Remove a model",
            "",
            "**Profile Model Management (owner only):**",
            "`;add pmodel <name>` - Create new profile model",
            "  - Interactive setup for base_model, sys_prompt, etc.",
            "`;edit pmodel <name> <field> <value>` - Edit profile model",
            "  - fields: base_model, sys_prompt, cost, level, is_live",
            "`;show pmodels` - List all profile models"
        ]

    await ctx.send("\n".join(lines), allowed_mentions=discord.AllowedMentions.none())


async def getid_cmd(ctx: commands.Context, member: discord.Member = None):
    if member is None:
        await ctx.send(f"Your ID: {ctx.author.id}", allowed_mentions=discord.AllowedMentions.none())
    else:
        await ctx.send(f"{member} ID: {member.id}", allowed_mentions=discord.AllowedMentions.none())

async def auth_cmd(ctx: commands.Context, id_or_mention: str):
    global _authorized_users
    uid = _extract_user_id_from_str(id_or_mention)
    if uid is None:
        await ctx.send("Invalid parameter. Provide a user ID or a mention.", allowed_mentions=discord.AllowedMentions.none())
        return

    if uid in _authorized_users:
        await ctx.send(f"ID {uid} is already authorized.", allowed_mentions=discord.AllowedMentions.none())
        return

    success = add_authorized_user(uid)
    if success:
        await ctx.send(f"Added ID {uid} to authorized list.", allowed_mentions=discord.AllowedMentions.none())
    else:
        await ctx.send(f"Failed to add ID {uid} to authorized list.", allowed_mentions=discord.AllowedMentions.none())

async def deauth_cmd(ctx: commands.Context, id_or_mention: str):
    global _authorized_users
    uid = _extract_user_id_from_str(id_or_mention)
    if uid is None:
        await ctx.send("Invalid parameter. Provide a user ID or a mention.", allowed_mentions=discord.AllowedMentions.none())
        return

    if uid not in _authorized_users:
        await ctx.send(f"ID {uid} is not in the authorized list.", allowed_mentions=discord.AllowedMentions.none())
        return

    success = remove_authorized_user(uid)
    if success:
        await ctx.send(f"Removed ID {uid} from authorized list.", allowed_mentions=discord.AllowedMentions.none())
    else:
        await ctx.send(f"Failed to remove ID {uid} from authorized list.", allowed_mentions=discord.AllowedMentions.none())

async def ping_cmd(ctx: commands.Context):
    import time
    start_time = time.perf_counter()
    message = await ctx.send("Pinging...", allowed_mentions=discord.AllowedMentions.none())
    end_time = time.perf_counter()
    
    # Calculate response time (ms)
    latency_ms = round((end_time - start_time) * 1000)
    
    # Get bot's WebSocket latency (if available)
    ws_latency = round(_bot.latency * 1000) if _bot.latency else "N/A"
    
    # Update message with detailed info
    content = f"Pong! \nResponse: {latency_ms} ms\nWebSocket: {ws_latency} ms"
    await message.edit(content=content, allowed_mentions=discord.AllowedMentions.none())

# ------------------------------------------------------------------
# Owner‑only memory commands
# ------------------------------------------------------------------
async def memory_cmd(ctx: commands.Context, member: discord.Member = None):
    """View the conversation history of *member* (or the author)."""
    target = member or ctx.author
    if _memory_store is None:
        await ctx.send("Memory feature not initialized.", allowed_mentions=discord.AllowedMentions.none())
        return

    mem = _memory_store.get_user_messages(target.id)
    if not mem:
        await ctx.send(f"No memory for {target}.", allowed_mentions=discord.AllowedMentions.none())
        return

    lines = []
    for i, msg in enumerate(mem[-10:], start=1):
        content = msg["content"]
        preview = (content[:120] + "…") if len(content) > 120 else content
        lines.append(f"{i:02d}. **{msg['role']}**: {preview}")

    await ctx.send("\n".join(lines), allowed_mentions=discord.AllowedMentions.none())


async def clearmemory_cmd(ctx: commands.Context, target: discord.Member = None):
    """Owner‑only: delete the conversation history of *target* (or the author)."""
    target = target or ctx.author
    if _memory_store is None:
        await ctx.send("Memory feature not initialized.", allowed_mentions=discord.AllowedMentions.none())
        return

    _memory_store.clear_user(target.id)
    await ctx.send(f"Cleared memory for {target}.", allowed_mentions=discord.AllowedMentions.none())

# ------------------------------------------------------------------
# NEW: Add command dispatcher (owner only)
# ------------------------------------------------------------------
async def add_cmd(ctx: commands.Context, resource_type: str = None, *, value: str = None):
    try:
        is_owner = await _bot.is_owner(ctx.author)
    except Exception:
        is_owner = False
        
    if not is_owner:
        await ctx.send("This command is only available to the bot owner.", 
                      allowed_mentions=discord.AllowedMentions.none())
        return

    if resource_type is None:
        await ctx.send("Usage:\n`;add model <model_name> <credit_cost> <access_level>`\n`;add credit @user <amount>`\n`;add pmodel <name>`", 
                      allowed_mentions=discord.AllowedMentions.none())
        return
    
    resource_type = resource_type.lower()
    
    if resource_type == "pmodel":
        if not value:
            await ctx.send("Please specify a profile model name.", allowed_mentions=discord.AllowedMentions.none())
            return
            
        name = value.strip()
        success, message = _mongodb_store.add_profile_model(name)
        await ctx.send(message, allowed_mentions=discord.AllowedMentions.none())
        return

    if resource_type == "model":
        if not value:
            await ctx.send("Usage: `;add model <model_name> <credit_cost> <access_level>`\nExample: `;add model gpt-6 5 2`", 
                          allowed_mentions=discord.AllowedMentions.none())
            return
            
        # Parse arguments
        try:
            args = value.split()
            if len(args) != 3:
                raise ValueError("Need model_name, credit_cost and access_level")
                
            model_name = args[0].strip()
            credit_cost = int(args[1])
            access_level = int(args[2])
            
            if not model_name:
                raise ValueError("Model name cannot be empty")
                
            if credit_cost < 0:
                raise ValueError("Credit cost must be positive")
                
            if access_level not in [0, 1, 2, 3]:
                raise ValueError("Access level must be 0, 1, 2, or 3")
                
        except ValueError as e:
            await ctx.send(f"Error: {str(e)}", allowed_mentions=discord.AllowedMentions.none())
            return
            
        # Check if MongoDB is enabled
        if not _config.USE_MONGODB:
            await ctx.send("Model management requires MongoDB mode to be enabled.", 
                          allowed_mentions=discord.AllowedMentions.none())
            return
        
        success, message = _mongodb_store.add_supported_model(model_name, credit_cost, access_level)
        await ctx.send(message, allowed_mentions=discord.AllowedMentions.none())
        
    elif resource_type == "credit":
        if not value:
            await ctx.send("Usage: `;add credit @user <amount>`", 
                          allowed_mentions=discord.AllowedMentions.none())
            return
            
        # Parse arguments
        try:
            args = value.split()
            if len(args) != 2:
                raise ValueError("Need both user mention and amount")
                
            # Get user from mention
            target_user = await commands.MemberConverter().convert(ctx, args[0])
            amount = int(args[1])
            
            if amount <= 0:
                raise ValueError("Amount must be positive")
                
        except (ValueError, commands.MemberNotFound) as e:
            await ctx.send(f"Error: {str(e)}", allowed_mentions=discord.AllowedMentions.none())
            return
            
        if _use_mongodb_auth:
            success, new_balance = _mongodb_store.add_user_credit(target_user.id, amount)
            if success:
                await ctx.send(f"Added {amount} credits to {target_user}'s balance. New balance: {new_balance}", 
                              allowed_mentions=discord.AllowedMentions.none())
            else:
                await ctx.send("Failed to add credits", 
                              allowed_mentions=discord.AllowedMentions.none())
        else:
            await ctx.send("Credit system requires MongoDB mode", 
                          allowed_mentions=discord.AllowedMentions.none())
    elif resource_type == "pmodel":
        if not value:
            await ctx.send("Please specify a profile model name.", allowed_mentions=discord.AllowedMentions.none())
            return
            
        name = value.strip()
        
        # Interactive setup
        await ctx.send("Please enter the base model name:", allowed_mentions=discord.AllowedMentions.none())
        try:
            base_model_msg = await bot.wait_for(
                'message',
                check=lambda m: m.author == ctx.author and m.channel == ctx.channel,
                timeout=30.0
            )
            base_model = base_model_msg.content.strip()
            
            await ctx.send("Please enter the system prompt:", allowed_mentions=discord.AllowedMentions.none())
            sys_prompt_msg = await bot.wait_for(
                'message', 
                check=lambda m: m.author == ctx.author and m.channel == ctx.channel,
                timeout=60.0
            )
            sys_prompt = sys_prompt_msg.content
            
            await ctx.send("Enter credit cost (default: 0):", allowed_mentions=discord.AllowedMentions.none())
            cost_msg = await bot.wait_for(
                'message',
                check=lambda m: m.author == ctx.author and m.channel == ctx.channel,
                timeout=30.0
            )
            try:
                cost = int(cost_msg.content)
            except:
                cost = 0
                
            await ctx.send("Enter access level (0-3, default: 0):", allowed_mentions=discord.AllowedMentions.none())
            level_msg = await bot.wait_for(
                'message',
                check=lambda m: m.author == ctx.author and m.channel == ctx.channel,
                timeout=30.0
            )
            try:
                level = int(level_msg.content)
                if level not in [0,1,2,3]:
                    level = 0
            except:
                level = 0
                
            await ctx.send("Enable live preview? (yes/no, default: no):", allowed_mentions=discord.AllowedMentions.none())
            live_msg = await bot.wait_for(
                'message',
                check=lambda m: m.author == ctx.author and m.channel == ctx.channel,
                timeout=30.0
            )
            is_live = live_msg.content.lower() in ['yes', 'y', 'true', '1']
            
            success, message = _mongodb_store.add_profile_model(
                name=name,
                base_model=base_model,
                sys_prompt=sys_prompt,
                credit_cost=cost,
                access_level=level,
                is_live=is_live
            )
            
            await ctx.send(message, allowed_mentions=discord.AllowedMentions.none())
            
        except asyncio.TimeoutError:
            await ctx.send("Setup timed out. Please try again.", allowed_mentions=discord.AllowedMentions.none())
    else:
        await ctx.send(f"Unknown resource type '{resource_type}'. Available: `model` or `credit`", 
                      allowed_mentions=discord.AllowedMentions.none())

# ------------------------------------------------------------------
# NEW: Remove command dispatcher (owner only)
# ------------------------------------------------------------------
async def remove_cmd(ctx: commands.Context, resource_type: str = None, *, value: str = None):
    """Remove command dispatcher - handles: remove model <model_name>"""
    # Check if user is owner
    try:
        is_owner = await _bot.is_owner(ctx.author)
    except Exception:
        is_owner = False
        
    if not is_owner:
        await ctx.send("This command is only available to the bot owner.", 
                      allowed_mentions=discord.AllowedMentions.none())
        return
    
    if resource_type is None:
        await ctx.send("Usage: `;remove model <model_name>`", 
                      allowed_mentions=discord.AllowedMentions.none())
        return
    
    resource_type = resource_type.lower()
    
    if resource_type == "model":
        if value is None:
            await ctx.send("Please specify a model name. Example: `;remove model old-model`", 
                          allowed_mentions=discord.AllowedMentions.none())
            return
        
        model_name = value.strip()
        if not model_name:
            await ctx.send("Model name cannot be empty.", 
                          allowed_mentions=discord.AllowedMentions.none())
            return
        
        # Check if MongoDB is enabled
        if not _config.USE_MONGODB:
            await ctx.send("Model management requires MongoDB mode to be enabled.", 
                          allowed_mentions=discord.AllowedMentions.none())
            return
        
        success, message = _mongodb_store.remove_supported_model(model_name)
        await ctx.send(message, allowed_mentions=discord.AllowedMentions.none())
        
    else:
        await ctx.send(f"Unknown resource type '{resource_type}'. Available: `model`", 
                      allowed_mentions=discord.AllowedMentions.none())

# ------------------------------------------------------------------
# NEW: Edit command dispatcher (owner only)
# ------------------------------------------------------------------
async def edit_cmd(ctx: commands.Context, resource_type: str = None, *, value: str = None):
    """Edit command dispatcher - handles: edit model <model_name> <credit_cost> <access_level>"""
    try:
        is_owner = await _bot.is_owner(ctx.author)
    except Exception:
        is_owner = False
        
    if not is_owner:
        await ctx.send("This command is only available to the bot owner.", 
                      allowed_mentions=discord.AllowedMentions.none())
        return
    
    if resource_type is None:
        await ctx.send("Usage:\n`;edit model <model_name> <credit_cost> <access_level>`", 
                      allowed_mentions=discord.AllowedMentions.none())
        return
    
    resource_type = resource_type.lower()
    
    if resource_type == "model":
        if not value:
            await ctx.send("Usage: `;edit model <model_name> <credit_cost> <access_level>`\nExample: `;edit model gpt-6 10 2`", 
                          allowed_mentions=discord.AllowedMentions.none())
            return
            
        # Parse arguments
        try:
            args = value.split()
            if len(args) != 3:
                raise ValueError("Need model_name, credit_cost and access_level")
                
            model_name = args[0].strip()
            credit_cost = int(args[1])
            access_level = int(args[2])
            
            if not model_name:
                raise ValueError("Model name cannot be empty")
                
            if credit_cost < 0:
                raise ValueError("Credit cost must be positive")
                
            if access_level not in [0, 1, 2, 3]:
                raise ValueError("Access level must be 0, 1, 2, or 3")
                
        except ValueError as e:
            await ctx.send(f"Error: {str(e)}", allowed_mentions=discord.AllowedMentions.none())
            return
            
        # Check if MongoDB is enabled
        if not _config.USE_MONGODB:
            await ctx.send("Model management requires MongoDB mode to be enabled.", 
                          allowed_mentions=discord.AllowedMentions.none())
            return
        
        success, message = _mongodb_store.edit_supported_model(model_name, credit_cost, access_level)
        await ctx.send(message, allowed_mentions=discord.AllowedMentions.none())
        
    elif resource_type == "pmodel":
        if not value:
            await ctx.send("Usage: `;edit pmodel <name> <field> <value>`", allowed_mentions=discord.AllowedMentions.none())
            return
            
        try:
            args = value.split(maxsplit=2)
            if len(args) < 3:
                raise ValueError("Need name, field and value")
                
            name = args[0].strip()
            field = args[1].strip().lower()
            new_value = args[2].strip()
            
            success, message = _mongodb_store.edit_profile_model(name, field, new_value)
            await ctx.send(message, allowed_mentions=discord.AllowedMentions.none())
            
        except ValueError as e:
            await ctx.send(str(e), allowed_mentions=discord.AllowedMentions.none())
            
    else:
        await ctx.send(f"Unknown resource type '{resource_type}'. Available: `model`", 
                      allowed_mentions=discord.AllowedMentions.none())

# ------------------------------------------------------------------
# Set command dispatcher (updated)
# ------------------------------------------------------------------
async def set_cmd(ctx: commands.Context, attribute: str = None, *, value: str = None):
    """Set command dispatcher - handles: set model <model>, set sys_prompt <prompt>, set level @user <level>"""
    # Check authorization
    if not await is_authorized_user(ctx.author):
        await ctx.send("You do not have permission to use this command.", 
                      allowed_mentions=discord.AllowedMentions.none())
        return
    
    if attribute is None:
        await ctx.send("Usage:\n`;set model <model>`\n`;set sys_prompt <prompt>`\n`;set level @user <level>`", 
                      allowed_mentions=discord.AllowedMentions.none())
        return
    
    attribute = attribute.lower()
    
    if attribute == "model":
        if value is None:
            # Get both regular and profile models
            all_models = []
            if _use_mongodb_auth:
                # Get regular models
                regular_models = _mongodb_store.list_all_models()
                all_models.extend([m.get("model_name") for m in regular_models])
                
                # Get profile models
                profile_models = _mongodb_store.list_profile_models()
                all_models.extend([p.get("name") for p in profile_models])
            else:
                all_models = list(_user_config_manager.get_supported_models())
                
            supported_list = ", ".join(sorted(all_models))
            await ctx.send(f"Please specify a model. Example: `;set model gpt-oss-120b`\n**Available models:** {supported_list}", 
                          allowed_mentions=discord.AllowedMentions.none())
            return

        # Check if value is a profile model first
        profile_model = None
        if _use_mongodb_auth:
            profile_model = _mongodb_store.get_profile_model(value.strip())

        if profile_model:
            success, message = _user_config_manager.set_user_model(ctx.author.id, value.strip())
            await ctx.send(f"Successfully set model to profile: {value.strip()}", allowed_mentions=discord.AllowedMentions.none())
            return
        else:
            # Regular model handling
            available, error = _call_api.is_model_available(value.strip())
            if not available:
                await ctx.send(error, allowed_mentions=discord.AllowedMentions.none())
                return

            success, message = _user_config_manager.set_user_model(ctx.author.id, value.strip())
            await ctx.send(message, allowed_mentions=discord.AllowedMentions.none())

    elif attribute == "sys_prompt":
        if value is None:
            await ctx.send("Please provide a system prompt. Example: `;set sys_prompt You are a helpful AI assistant`", 
                          allowed_mentions=discord.AllowedMentions.none())
            return
        
        success, message = _user_config_manager.set_user_system_prompt(ctx.author.id, value)
        await ctx.send(message, allowed_mentions=discord.AllowedMentions.none())
        
    elif attribute == "level":
        # Check if user is owner
        try:
            is_owner = await _bot.is_owner(ctx.author)
        except Exception:
            is_owner = False
            
        if not is_owner:
            await ctx.send("Only the bot owner can set user levels.", 
                          allowed_mentions=discord.AllowedMentions.none())
            return
            
        if not value:
            await ctx.send("Usage: `;set level @user <level>` (0=Basic, 1=Advanced, 2=Premium, 3=Ultimate)", 
                          allowed_mentions=discord.AllowedMentions.none())
            return
            
        # Parse arguments
        try:
            args = value.split()
            if len(args) != 2:
                raise ValueError("Need both user mention and level")
                
            # Get user from mention
            target_user = await commands.MemberConverter().convert(ctx, args[0])
            level = int(args[1])
            
            if level not in [0, 1, 2, 3]:
                raise ValueError("Level must be 0, 1, 2, or 3")
                
        except (ValueError, commands.MemberNotFound) as e:
            await ctx.send(f"Error: {str(e)}", allowed_mentions=discord.AllowedMentions.none())
            return
            
        if _use_mongodb_auth:
            success = _mongodb_store.set_user_level(target_user.id, level)
            if success:
                level_names = {0: "Basic", 1: "Advanced", 2: "Premium", 3: "Ultimate"}
                await ctx.send(f"Set {target_user}'s level to {level_names[level]} (Level {level})", 
                              allowed_mentions=discord.AllowedMentions.none())
            else:
                await ctx.send("Failed to set user level", 
                              allowed_mentions=discord.AllowedMentions.none())
        else:
            await ctx.send("User levels require MongoDB mode", 
                          allowed_mentions=discord.AllowedMentions.none())
            
    else:
        await ctx.send(f"Unknown attribute '{attribute}'. Use: `model`, `sys_prompt`, or `level`", 
                      allowed_mentions=discord.AllowedMentions.none())

# ------------------------------------------------------------------
# Show command dispatcher (updated)
# ------------------------------------------------------------------
async def show_cmd(ctx: commands.Context, item: str = None, detail_or_user: str = None):
    if item is None:
        await ctx.send("Usage: `;show profile`, `;show sys_prompt [@user]`, `;show model`, `;show pmodel [name]`", 
                      allowed_mentions=discord.AllowedMentions.none())
        return
    
    item = item.lower()
    
    # Add pmodel handler before profile handler
    if item == "pmodel":
        if detail_or_user:
            # Show specific profile model
            success, message = _mongodb_store.get_profile_model_details(detail_or_user)
            await ctx.send(message, allowed_mentions=discord.AllowedMentions.none())
        else:
            # List all profile models
            all_profiles = _mongodb_store.list_profile_models()
            if all_profiles:
                lines = ["**Profile Models:**"]
                for p in all_profiles:
                    name = p.get("name", "Unknown")
                    base = p.get("base_model", "Not set")
                    cost = p.get("credit_cost", 0)
                    level = p.get("access_level", 0)
                    is_live = "✅" if p.get("is_live") else "❌"
                    lines.append(f"• `{name}` -> {base} (Cost: {cost}, Level: {level}, Live: {is_live})")
            else:
                lines = ["No profile models found."]
            await ctx.send("\n".join(lines), allowed_mentions=discord.AllowedMentions.none())
        return

    elif item == "profile":
        # The profile command is public: **anyone** can invoke it.
        # Optional target member: 1st argument if present.
        target_user: Optional[discord.Member] = None
        if detail_or_user:
            target_arg = detail_or_user
            try:
                # Try to resolve a Member from the argument
                target_user = await commands.MemberConverter().convert(ctx, target_arg)
            except Exception:
                # Maybe it's a raw ID
                try:
                    uid = int(target_arg)
                    guild = ctx.guild
                    if guild:
                        target_user = await guild.fetch_member(uid)
                except Exception:
                    pass

        else:
            # No target specified => show own profile
            target_user = ctx.author

        if target_user is None:
            await ctx.send(
                f"Could not find member `{target_arg}`.",
                allowed_mentions=discord.AllowedMentions.none())
            return

        if target_user != ctx.author:
            try:
                is_owner = await _bot.is_owner(ctx.author)
            except Exception:
                is_owner = False

            if not is_owner:
                await ctx.send(
                    "You can only view your own profile.",
                    allowed_mentions=discord.AllowedMentions.none())
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

        # Build profile display without system prompt
        lines = [
            f"**👤 Profile for {target_user}:**",
            f"🤖 **Current Model**: {model}",
            f"💰 **Credit Balance**: {credit}",
            f"🔑 **Access Level**: {level_desc}",
            "",
            "Use `;show sys_prompt` to view system prompt."
        ]

        await ctx.send("\n".join(lines), allowed_mentions=discord.AllowedMentions.none())
        return

    elif item == "sys_prompt":
        # Handle system prompt display
        target_user: Optional[discord.Member] = None
        if detail_or_user:
            try:
                target_user = await commands.MemberConverter().convert(ctx, detail_or_user)
            except Exception:
                await ctx.send("Could not find specified user.", allowed_mentions=discord.AllowedMentions.none())
                return
            
            # Check if viewer is owner when viewing other's prompt
            if target_user != ctx.author:
                try:
                    is_owner = await _bot.is_owner(ctx.author)
                except Exception:
                    is_owner = False

                if not is_owner:
                    await ctx.send("Only the bot owner can view other users' system prompts.",
                                 allowed_mentions=discord.AllowedMentions.none())
                    return
        else:
            target_user = ctx.author
            
        # Get system prompt
        user_config = _user_config_manager.get_user_config(target_user.id)
        prompt = user_config["system_prompt"]
        
        # Format display
        lines = [   
            f"**📝 System Prompt for {target_user}:**",
            "```",
            prompt,
            "```",
            "",
            "This is the prompt used to guide the AI's responses."
        ]
        
        await ctx.send("\n".join(lines), allowed_mentions=discord.AllowedMentions.none())
        return

    # ---------- Handle `model` / `models` ----------
    elif item in {"model", "models"}:
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
                    combined_models.append({
                        "name": model.get("model_name", "Unknown"),
                        "credit_cost": model.get("credit_cost", 0),
                        "access_level": model.get("access_level", 0),
                    })
                
                # Add profile models
                for pmodel in all_pmodels:
                    combined_models.append({
                        "name": pmodel.get("name", "Unknown"),
                        "credit_cost": pmodel.get("credit_cost", 0),
                        "access_level": pmodel.get("access_level", 0),
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
                    level_names = {0: "Basic", 1: "Advanced", 2: "Premium", 3: "Ultimate"}
                    level_name = level_names.get(access_level, f"Level {access_level}")
                    
                    # Add level header if level changed
                    if current_level != access_level:
                        models_info.append(f"\n**{level_name} Models:**")
                        current_level = access_level
                        
                    models_info.append(f"• `{model_name}` - {credit_cost} credits")

                lines = [
                    "**Available AI Models:**",
                    *models_info,
                    "",
                    "Use `;set model <model_name>` to change your model."
                ]
            else:
                lines = ["No models found."]

        else:
            # Fallback for file mode
            supported_models = _user_config_manager.get_supported_models()
            models_list = "\n".join(f"• `{model}`" for model in sorted(supported_models))
            lines = [
                "**Supported AI Models:**",
                models_list,
                "",
                "Use `;set model <model_name>` to change your model."
            ]

        await ctx.send("\n".join(lines), allowed_mentions=discord.AllowedMentions.none())

    # ---------- Handle `auth` ----------
    elif item == "auth":
        try:
            is_owner = await _bot.is_owner(ctx.author)
        except Exception:
            is_owner = False

        if not is_owner:
            await ctx.send(
                "Only the bot owner can view the authorized users list.",
                allowed_mentions=discord.AllowedMentions.none())
            return

        if not _authorized_users:
            await ctx.send("Authorized users list is empty.", allowed_mentions=discord.AllowedMentions.none())
            return

        body = "\n".join(str(x) for x in sorted(_authorized_users))
        if len(body) > 1900:
            if _use_mongodb_auth:
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                    f.write("Authorized Users:\n")
                    f.write(body)
                    temp_path = f.name
                try:
                    await ctx.send(
                        "Too long data, sending authorized_users.txt file.",
                        allowed_mentions=discord.AllowedMentions.none(),
                        file=discord.File(temp_path, filename="authorized_users.txt"))
                finally:
                    Path(temp_path).unlink(missing_ok=True)
            else:
                fp = _config.AUTHORIZED_STORE if _config.AUTHORIZED_STORE.exists() else None
                if fp:
                    await ctx.send(
                        "Too long data, sending authorized.json file.",
                        allowed_mentions=discord.AllowedMentions.none(),
                        file=discord.File(fp))
                else:
                    await ctx.send(
                        "Too long data, but no authorized.json file found.",
                        allowed_mentions=discord.AllowedMentions.none())
        else:
            await ctx.send(f"**Authorized users list:**\n{body}",
                           allowed_mentions=discord.AllowedMentions.none())
    elif item == "pmodels":
        all_profiles = _mongodb_store.list_profile_models()
        if all_profiles:
            lines = ["**Profile Models:**"]
            for p in all_profiles:
                name = p.get("name", "Unknown")
                base = p.get("base_model", "Unknown")
                cost = p.get("credit_cost", 0)
                level = p.get("access_level", 0)
                is_live = "✅" if p.get("is_live") else "❌"
                lines.append(f"• `{name}` -> {base} (Cost: {cost}, Level: {level}, Live: {is_live})")
        else:
            lines = ["No profile models found."]
            
        await ctx.send("\n".join(lines), allowed_mentions=discord.AllowedMentions.none())
        
    else:
        await ctx.send(
            f"Unknown item {item}. Use `config`, `model`, `models detailed` (owner) or `auth` (owner).",
            allowed_mentions=discord.AllowedMentions.none())
# ------------------------------------------------------------------
# AI Request Processing Function (used by queue)
# ------------------------------------------------------------------
# Replace your process_ai_request function in functions.py with this fixed version:

# Fixed process_ai_request function - replace in functions.py

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

def get_vietnam_timestamp() -> str:
    """Get current timestamp in GMT+7 (Vietnam timezone)"""
    vietnam_tz = timezone(timedelta(hours=7))
    now = datetime.now(vietnam_tz)
    
    formatted_time = now.strftime("%A, %B %d, %Y - %H:%M:%S")
    return f"Thời gian hiện tại: {formatted_time} (GMT+7) : "
    
async def process_ai_request(request):
    """Process a single AI request from the queue with stream support"""
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
        
        # Build message payload
        payload_messages = [user_system_message]
        if _memory_store:
            payload_messages.extend(_memory_store.get_user_messages(user_id))
            
            final_user_text = f"{get_vietnam_timestamp()}{final_user_text}"
        payload_messages.append({"role": "user", "content": final_user_text})

        # Create initial response message
        response_msg = await message.channel.send(
            "...",
            reference=message,
            allowed_mentions=discord.AllowedMentions.none()
        )

        # Stream or regular response
        if is_live:
            collected_response = ""
            last_update = 0
            
            # Process stream
            async for chunk in _call_api.call_openai_proxy_stream(payload_messages, user_model):
                collected_response += chunk
                
                # Update message every 1 second or when chunk is large
                current_time = time.time()
                if (current_time - last_update > 1 or len(chunk) > 100) and collected_response:
                    formatted_response = convert_latex_to_discord(collected_response)
                    try:
                        await response_msg.edit(content=formatted_response)
                        last_update = current_time
                    except Exception as e:
                        logger.error(f"Failed to update stream message: {e}")
                        
            # Final update and process completion
            if collected_response:
                final_response = convert_latex_to_discord(collected_response)
                await response_msg.edit(content=final_response)
                
                # Save to memory store
                if _memory_store:
                    _memory_store.add_message(user_id, {"role": "user", "content": final_user_text})
                    _memory_store.add_message(user_id, {"role": "assistant", "content": collected_response})
                    
                # Deduct credits after successful completion
                if _use_mongodb_auth and model_info:
                    await deduct_credits(user_id, model_info.get("credit_cost", 0))
                    
        else:
            # Regular non-streaming call - existing code remains unchanged
            ok, resp = await asyncio.get_event_loop().run_in_executor(
                None,
                _call_api.call_openai_proxy,
                payload_messages,
                user_model
            )
            
            if ok and resp:
                formatted_response = convert_latex_to_discord(resp)
                await response_msg.edit(content=formatted_response)
                
                # Save to memory store
                if _memory_store:
                    _memory_store.add_message(user_id, {"role": "user", "content": final_user_text})
                    _memory_store.add_message(user_id, {"role": "assistant", "content": resp})
                
                # Deduct credits after successful completion    
                if _use_mongodb_auth and model_info:
                    await deduct_credits(user_id, model_info.get("credit_cost", 0))
            else:
                error_msg = resp or "Unknown error"
                await response_msg.edit(content=f"❌ Error: {error_msg}")

    except Exception as e:
        logger.exception(f"Error in request processing for user {user_id}")
        await message.channel.send(
            f"⚠️ Internal error: {e}",
            reference=message,
            allowed_mentions=discord.AllowedMentions.none()
        )

# Add helper functions
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

# ------------------------------------------------------------------
# Fixed Message formatting helpers with proper table handling
# ------------------------------------------------------------------

def convert_latex_to_discord(text: str) -> str:
    """
    Fixed version - only protect code blocks, NOT markdown tables
    """
    
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

def split_message_smart(text: str, max_length: int = 2000) -> list[str]:
    """
    Smart message splitting that keeps tables intact
    """
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
            else:
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
                    # Table is too large - need to split it intelligently
                    # Keep header + separator together if possible
                    header_lines = []
                    data_lines = []
                    
                    # Try to identify header vs data
                    for j, tline in enumerate(table_lines):
                        if j < 2:  # Usually header + separator
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
                            
                            current_chunk = current_table_chunk
                        else:
                            # Even header is too long, fallback to line by line
                            for tline in table_lines:
                                test_line = current_chunk + ('\n' if current_chunk else '') + tline
                                if len(test_line) <= max_length:
                                    current_chunk = test_line
                                else:
                                    if current_chunk:
                                        chunks.append(current_chunk)
                                    current_chunk = tline
                    else:
                        # Fallback: process line by line
                        for tline in table_lines:
                            test_line = current_chunk + ('\n' if current_chunk else '') + tline
                            if len(test_line) <= max_length:
                                current_chunk = test_line
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk)
                                current_chunk = tline
            
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
                # Single line is too long - split it
                if len(line) > max_length:
                    # Preserve indentation
                    leading_whitespace = re.match(r'^(\s*)', line).group(1)
                    line_content = line[len(leading_whitespace):]
                    
                    while len(line_content) > max_length - len(leading_whitespace):
                        available_space = max_length - len(leading_whitespace)
                        part_content = line_content[:available_space]
                        chunks.append(leading_whitespace + part_content)
                        line_content = line_content[available_space:]
                    
                    if line_content:
                        current_chunk = leading_whitespace + line_content
                else:
                    current_chunk = line
        else:
            current_chunk = test_chunk
        
        i += 1
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Update the message sending functions
async def send_long_message(channel, content: str, max_msg_length: int = 2000):
    """
    Send long message with proper table handling
    """
    # First apply LaTeX conversion (which now preserves tables)
    formatted_content = convert_latex_to_discord(content)
    
    if len(formatted_content) <= max_msg_length:
        await channel.send(formatted_content, allowed_mentions=discord.AllowedMentions.none())
        return
    
    chunks = split_message_smart(formatted_content, max_msg_length)
    
    for i, chunk in enumerate(chunks):
        if i > 0:  # Add delay between messages
            await asyncio.sleep(0.3)
        await channel.send(chunk, allowed_mentions=discord.AllowedMentions.none())


async def send_long_message_with_reference(channel, content: str, reference_message: discord.Message, max_msg_length: int = 2000):
    """
    Send long message with reference and proper table handling
    """
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
        if i > 0:  # Add delay between messages
            await asyncio.sleep(0.3)
        
        # Only reference the original message for the first chunk
        ref = reference_message if i == 0 else None
        await channel.send(
            chunk,
            reference=ref,
            allowed_mentions=discord.AllowedMentions.none()
        )

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

    # 1️⃣ FIXED: Commands that start with prefix - DON'T manually process them here
    # Discord.py will automatically handle them via the command framework
    if content.startswith(";"):
        # Just return, let discord.py handle the command processing
        return

    # 2️⃣ Default trigger (DM or mention) - for AI responses
    authorized = await is_authorized_user(message.author)
    attachments = list(message.attachments or [])

    if not should_respond_default(message):
        # Not a DM or mention, let discord.py process any commands if present
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
    # Handle attachments
    # ------------------------------------------------------------------
    attachment_text = ""
    if attachments:
        files_info = await _read_attachments_as_text(attachments)
        attach_summary = []
        for fi in files_info:
            if fi.get("skipped"):
                attach_summary.append(f"- {fi['filename']}: SKIPPED ({fi.get('reason')})")
            else:
                attach_summary.append(f"- {fi['filename']}: included ({len(fi['text'])} chars)")
        header = "\n".join(attach_summary) + "\n\n"

        files_combined = ""
        for fi in files_info:
            if not fi.get("skipped"):
                files_combined += f"Filename: {fi['filename']}\n---\n{fi['text']}\n\n"
        attachment_text = header + files_combined

    final_user_text = (attachment_text + user_text).strip()
    if not final_user_text:
        await message.channel.send(
            "Please send a message (mention me or DM me) with your question.",
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

# ------------------------------------------------------------------
# Setup – register commands, listeners, load data (updated for MongoDB)
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
    # Remove default help (if any)
    # ------------------------------------------------------------------
    try:
        bot.remove_command("help")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Register commands (idempotent – duplicates are harmless)
    # ------------------------------------------------------------------
    bot.add_command(commands.Command(help_cmd, name="help"))
    bot.add_command(commands.Command(getid_cmd, name="getid"))
    bot.add_command(commands.Command(ping_cmd, name="ping"))

    # Set and Show commands
    bot.add_command(commands.Command(set_cmd, name="set"))
    bot.add_command(commands.Command(show_cmd, name="show"))

    # Owner commands
    owner_check = commands.is_owner()
    bot.add_command(commands.Command(auth_cmd, name="auth", checks=[owner_check]))
    bot.add_command(commands.Command(deauth_cmd, name="deauth", checks=[owner_check]))
    bot.add_command(commands.Command(memory_cmd, name="memory", checks=[owner_check]))
    bot.add_command(commands.Command(clearmemory_cmd, name="clearmemory", checks=[owner_check]))
    
    # Model management commands (owner only)
    bot.add_command(commands.Command(add_cmd, name="add", checks=[owner_check]))
    bot.add_command(commands.Command(remove_cmd, name="remove", checks=[owner_check]))
    bot.add_command(commands.Command(edit_cmd, name="edit", checks=[owner_check]))

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

    logger.info("Commands registered: %s", sorted(c.name for c in bot.commands))