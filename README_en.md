# Ryuuko — Discord LLM Bot

Version: v5.0

Ryuuko is a Discord bot written in Python that uses a Large Language Model (LLM) — for example OpenAI / Gemini / a compatible endpoint — to answer messages, handle commands, and optionally store user "memory" / configuration in MongoDB.

---

## Key features

- AI replies using an LLM (prompt + per-user system prompt).  
- Per-user configuration (model, system prompt) with two backends:
  - MongoDB (persistent) if `USE_MONGODB` is enabled.
  - File mode (fallback) saved at `config/user_config.json`.  
- Async request queue to avoid concurrent processing for the same user and basic rate-limiting.  
- Wrapper for calling LLM APIs (module `call_api.py`).  
- Text utilities (module `functions.py`) — e.g., LaTeX → Unicode conversion, code block protection, smart splitting that preserves Markdown tables, etc.  
- Logging and friendly error handling for slash and legacy commands.

---

## Requirements

- Python 3.10+ (3.11 recommended)  
- Pip to install dependencies (see `requirements.txt`)  
- A Discord Bot token (Discord Developer Portal)  
- An API key / endpoint for an LLM (OpenAI / Gemini / compatible)  
- (Optional) MongoDB if you want persistent memory/config storage  
- (Optional) Docker & docker-compose to run MongoDB in a container

---

## Quick install

```bash
# 1. Clone repo
git clone https://github.com/zvwgvx/Ryuuko.git
cd Ryuuko

# 2. Create virtual environment and install deps
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Prepare configuration (see Configuration section)
```

---

## Configuration (env / config.json)

The project reads configuration from `config.json`. You can place values in that file or provide equivalent environment/config. DO NOT commit secrets (Discord token, API key, connection strings).

Important variables (in `config.json` or equivalent):
- DISCORD_TOKEN — Discord bot token (required)  
- OPENAI_API_KEY — OpenAI API key (or equivalent key)  
- OPENAI_API_BASE — (optional, if using a custom base URL)  
- OPENAI_MODEL — (default can be overridden per-user)  
- CLIENT_GEMINI_API_KEY, OWNER_GEMINI_API_KEY — (if integrating Gemini)  
- USE_MONGODB — true/false (enable MongoDB mode)  
- MONGODB_CONNECTION_STRING — MongoDB URI (if used)  
- MONGODB_DATABASE_NAME — DB name (default: discord_openai_proxy)  
- REQUEST_TIMEOUT — (int) API call timeout  
- MAX_MSG — (int) maximum message length to handle

Example `.env` (DO NOT commit):
```text
DISCORD_TOKEN=bot-token-here
OPENAI_API_KEY=sk-...
USE_MONGODB=true
MONGODB_CONNECTION_STRING=mongodb://user:pass@host:27017
```

---

## Running

```bash
# From repo root
python ryuuko.py

# Or run main module
python src/main.py
```

Notes:
- `ryuuko.py` adds `src/` to `sys.path` and calls `main.bot.run(DISCORD_TOKEN)` — ensure the token is valid.  
- If using MongoDB, verify `MONGODB_CONNECTION_STRING` and accessibility.

---

## Docker (optional)

Minimal Dockerfile (DO NOT include secrets):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "ryuuko.py"]
```

Docker-compose example for MongoDB:
```yaml
version: '3.8'
services:
  mongo:
    image: mongo:6
    restart: unless-stopped
    ports:
      - 27017:27017
    volumes:
      - mongo_data:/data/db
volumes:
  mongo_data:
```

---

## Project layout & module responsibilities

- ryuuko.py — entrypoint/launcher (adds src to sys.path, imports main, starts bot)  
- requirements.txt — Python dependencies  
- config.json — sample/placeholder config (do not store real secrets)  
- src/  
  - main.py — bot definition, event handlers, command sync, graceful shutdown.  
  - load_config.py — read `config.json`, initialize global config variables and logger.  
  - call_api.py — LLM API wrapper.  
  - user_config.py — per-user config management (model, system prompt), supports MongoDB or file fallback.  
  - memory_store.py — in-memory memory store when not using MongoDB, with trimming rules.  
  - mongodb_store.py — persistent storage: user configs, memories, supported models.  
  - request_queue.py — async PriorityQueue to manage requests (prevents concurrent user requests, rate-limiting, worker).  
  - functions.py — text utilities, command registration (slash commands), on_message listener, and AI request processing.

---

## Commands (actual)

The slash commands registered by `functions.setup()` include:

- `/help` — show commands and brief guidance. (everyone)  
- `/getid [mention_or_user]` — show your ID or the mentioned user's ID. (everyone)  
- `/ping` — check bot responsiveness / latency. (everyone)

Configuration group (authorized users):
- `/set model <model>` — set user's preferred model.  
- `/set sys_prompt <prompt>` — set user's system prompt.

Show group (authorized users):
- `/show profile [user]` — display a user's configuration.  
- `/show sys_prompt [user]` — view a user's system prompt.  
- `/show models` — list supported models.

Memory-related:
- `/memory` — manage per-user memory (see code for details).  
- `/clearmemory [user]` — clear conversation history for a user (authorized).

`add`, `remove`, `edit` groups — manage resources (models/profiles) with subcommands.  
Owner-only:
- `/auth <user>` — add a user to the authorized list.  
- `/deauth <user>` — remove a user's authorization.

Note: Subcommands and precise parameter signatures are implemented in `src/functions.py`. I can extract and list them in detail on request.

---

## How it works (summary)

1. Bot receives a message or slash command from Discord (main.py → functions).  
2. Authorization and user config lookup (user_config.py).  
3. Request is queued in `request_queue` to ensure sequential processing and enforce rate-limiting.  
4. The queue worker calls `call_api.py` to send the prompt to the LLM and receive a response.  
5. `functions.py` formats/sanitizes the result and sends it back to the user.  
6. If MongoDB is enabled, memory and configs are persisted in `mongodb_store.py`.

---

## Debug & Troubleshooting

- Bot not online: check DISCORD_TOKEN, invite status, and permissions.  
- LLM errors: verify OPENAI_API_KEY / OPENAI_API_BASE / CLIENT_GEMINI_API_KEY.  
- MongoDB errors: verify MONGODB_CONNECTION_STRING, network/firewall, and credentials.  
- Rate-limiting / denied requests: inspect returned messages; the bot prevents rapid duplicate requests from the same user.  
- Check console logs for traces and exceptions.

---

## Community & Support

Join the official project Discord to discuss, ask questions, and get help:  
https://discord.gg/25mcSRMadU

---

## Security & operations

- DO NOT commit secrets (token, keys, connection strings). Use `.gitignore`.  
- For Docker deployments, pass secrets via environment variables or a secret manager.  
- Add indexes to MongoDB collections that are frequently queried (e.g., user_id, model names).  
- Add retry/backoff behavior for network operations (LLM, DB).

---

## Contributing

1. Fork the repository  
2. Create a branch: feature/describe-it  
3. Open a PR with detailed description  
4. Add unit tests for important logic (request_queue, user_config, mongodb_store)

---

## License

Default: MIT.  
This project is licensed under the MIT License — see the included `LICENSE` file for full terms.