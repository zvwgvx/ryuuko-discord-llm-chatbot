# Ryuuko — Discord Chatbot (Setup & Run)

**Ryuuko** is a Discord bot that uses a large language model (LLM) — e.g., OpenAI or a compatible endpoint — to respond to messages, handle commands, and optionally store user "memory" in MongoDB. This README explains how to set up, configure, run, and troubleshoot the project.

---

## Requirements

* Python 3.10 or newer (3.11 recommended)
* Git (to clone the repository)
* A Discord Bot token (from Discord Developer Portal)
* An LLM API key / endpoint (e.g., OpenAI API key or a custom compatible endpoint)
* (Optional) Docker & docker-compose if you want to run MongoDB in a container

---

## Project layout (important files)

```
Ryuuko/
  ├─ Ryuuko.py                # Launcher / entrypoint
  ├─ requirements.txt
  ├─ config.json             # Example configuration (placeholders)
  └─ src/
      ├─ main.py
      ├─ call_api.py
      ├─ load_config.py
      ├─ user_config.py
      ├─ memory_store.py
      ├─ mongodb_store.py
      ├─ request_queue.py
      └─ functions.py
```

---

## Step 1 — Clone & create virtual environment

Linux / macOS:

```bash
git clone https://github.com/zvwgvx/Ryuuko.git
cd Ryuuko
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
git clone https://github.com/zvwgvx/Ryuuko.git
cd Ryuuko
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Step 2 — Configuration (recommended: environment variables)

**Do not** commit secrets (Discord token, API keys, database credentials) into version control. The project uses a `config.json` file with placeholders — you should edit this file directly with your values. Make sure to add any real `config.json` containing secrets to `.gitignore`.

```
# .env.example (copy to .env and fill in values)
DISCORD_TOKEN=your_discord_bot_token
OPENAI_API_KEY=sk-xxxxx-or-other-key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-oss-120b

# Optional: enable MongoDB-backed memory
USE_MONGODB=true
MONGODB_CONNECTION_STRING=mongodb://user:pass@host:27017
MONGODB_DATABASE_NAME=discord_openai_proxy

# Optional runtime settings
REQUEST_TIMEOUT=100
MAX_MSG=1900
MEMORY_MAX_PER_USER=100
MEMORY_MAX_TOKENS=6000
```

### How env vars are used

`src/load_config.py` (or equivalent) reads configuration values. If you prefer `config.json`, use it for non-secret defaults only and let environment variables override sensitive values.

---

## Step 3 — Run the bot

After configuring environment variables (or editing `config.json`), start the bot:

```bash
python Ryuuko.py
# or if you prefer
python src/main.py
```

The bot should log in with the provided `DISCORD_TOKEN` and begin responding to messages on servers where it has been invited.

---

## Invite the bot to a server

1. Go to the Discord Developer Portal and open your application.
2. Under **OAuth2 → URL Generator**, select `bot` and `applications.commands` scopes.
3. Select the permissions your bot needs (start with minimal permissions and increase as necessary).
4. Copy the generated URL and open it in a browser to invite the bot to your server.

Alternatively, use this URL template (replace `CLIENT_ID` and `PERMISSIONS`):

```
https://discord.com/oauth2/authorize?client_id=CLIENT_ID&permissions=PERMISSIONS&scope=bot%20applications.commands
```

> **Security note:** avoid giving the bot more permissions than it needs. Do not use Administrator unless required.

---

## Optional: Run MongoDB locally (Docker)

If you enable `USE_MONGODB=true`, point `MONGODB_CONNECTION_STRING` to a reachable MongoDB instance. To run MongoDB locally with Docker:

`docker-compose.yml` example:

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

Start MongoDB:

```bash
docker-compose up -d
```

Then set `MONGODB_CONNECTION_STRING=mongodb://localhost:27017` (or include credentials if you enabled authentication).

---

## Example Dockerfile (optional)

If you want to containerize the bot, a minimal Dockerfile looks like this:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "Ryuuko.py"]
```

**Important:** do not bake secrets into the image. Pass environment variables at runtime using `docker run -e` or `docker-compose`.

---

## Troubleshooting & Debugging

* **Bot does not appear online / cannot log in**: Verify `DISCORD_TOKEN` is correct and the bot is invited to the server. Check that the bot is not disabled or banned on the server.
* **Missing permissions**: Ensure the bot has the necessary permissions in the server and for the commands you expect it to run.
* **LLM / OpenAI errors**: Verify `OPENAI_API_KEY` and `OPENAI_API_BASE`. If you use a custom endpoint, check the endpoint URL and any required headers.
* **MongoDB connection errors**: Confirm `MONGODB_CONNECTION_STRING`, network accessibility, firewall, and credentials.
* **Dependency / import errors**: Ensure you installed `requirements.txt` within the active virtual environment.
* **Check logs**: The bot uses logging. Inspect the console output for stack traces and errors. If you need more verbose logs, increase the logging level in the code.

If you still have trouble, consider sharing the relevant error logs (redact any secrets) so the maintainers can help.
