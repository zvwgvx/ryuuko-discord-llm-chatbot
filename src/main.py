#!/usr/bin/env python3
# coding: utf-8

import logging
import sys
import os
import signal
import asyncio

from discord.ext import commands
import discord

import load_config
import call_api
import functions
from request_queue import get_request_queue

# logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger("discord-openai-proxy.main")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix=";", intents=intents, help_command=None)

# Initialize functions module: register commands/listeners and load persisted data
functions.setup(bot, call_api, load_config)

# Graceful shutdown handler
async def shutdown_handler():
    """Handle graceful shutdown"""
    logger.info("Shutting down bot gracefully...")
    
    # Stop the request queue
    request_queue = get_request_queue()
    await request_queue.stop()
    
    # Close bot connection
    await bot.close()
    
    logger.info("Bot shutdown complete")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    asyncio.create_task(shutdown_handler())

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Optional: keep a simple on_ready here for early logging (functions.setup doesn't override)
@bot.event
async def on_ready():
    logger.info(f"Bot is ready: {bot.user} (id={bot.user.id}) pid={os.getpid()}")
    
    # Sync slash commands
    try:
        synced = await bot.tree.sync()
        logger.info(f"Synced {len(synced)} slash commands")
    except Exception as e:
        logger.exception(f"Failed to sync slash commands: {e}")
    
    # show registered commands
    try:
        cmds = sorted([c.name for c in bot.commands])
        logger.info("Registered legacy commands: %s", cmds)
        
        # Show slash commands
        slash_cmds = [cmd.name for cmd in bot.tree.get_commands()]
        logger.info("Registered slash commands: %s", slash_cmds)
    except Exception:
        logger.exception("Failed to list commands")

    # inspect on_message listeners
    try:
        listeners = list(getattr(bot, "_listeners", {}).get("on_message", []))
        logger.info("on_message listeners (count=%d): %s", len(listeners),
                    [f"{getattr(l,'__module__','?')}:{getattr(l,'__qualname__','?')} id={hex(id(l))}" for l in listeners])
    except Exception:
        logger.exception("Failed to inspect listeners")

# Add error handler for slash commands
@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: discord.app_commands.AppCommandError):
    """Handle slash command errors"""
    if isinstance(error, discord.app_commands.MissingPermissions):
        await interaction.response.send_message("⚠ Bạn không có quyền sử dụng lệnh này.", ephemeral=True)
        return
    
    if isinstance(error, discord.app_commands.CommandOnCooldown):
        await interaction.response.send_message(f"⚠ Lệnh đang trong thời gian chờ. Thử lại sau {error.retry_after:.1f} giây.", ephemeral=True)
        return
    
    logger.exception(f"Slash command error in {interaction.command}: {error}")
    
    try:
        if not interaction.response.is_done():
            await interaction.response.send_message("⚠ Đã xảy ra lỗi khi thực hiện lệnh.", ephemeral=True)
        else:
            await interaction.followup.send("⚠ Đã xảy ra lỗi khi thực hiện lệnh.", ephemeral=True)
    except:
        pass

# Add error handler for legacy commands (if any remain)
@bot.event
async def on_command_error(ctx, error):
    """Handle legacy command errors"""
    if isinstance(error, commands.CommandNotFound):
        return  # Ignore unknown commands
    
    if isinstance(error, commands.CheckFailure):
        await ctx.send("⚠ Bạn không có quyền sử dụng lệnh này.", allowed_mentions=discord.AllowedMentions.none())
        return
    
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"⚠ Thiếu tham số: {error.param}", allowed_mentions=discord.AllowedMentions.none())
        return
    
    logger.exception(f"Legacy command error in {ctx.command}: {error}")
    await ctx.send("⚠ Đã xảy ra lỗi khi thực hiện lệnh.", allowed_mentions=discord.AllowedMentions.none())

if __name__ == "__main__":
    try:
        bot.run(load_config.DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("Bot interrupted by user")
    except Exception:
        logger.exception("Bot exited with exception")
    finally:
        # Cleanup if needed
        logger.info("Main process exiting")