#!/usr/bin/env python3
# coding: utf-8

"""
Ryuuko.py – launcher for the Discord bot

The project layout is
    project/
        src/
            load_config.py
            call_api.py
            functions.py
            memory_store.py
            mongodb_store.py
            user_config.py
            request_queue.py
            main.py            ← contains the bot definition
        ryuuko.py              ← this file
        config.json
        .gitignore
        Readme.md
        requirements.txt
        

`ryuuko.py` simply makes *src* discoverable, imports the
`main` module that builds the bot, and starts the bot.

It keeps the same logging configuration already defined in
`src/main.py`; therefore the one-liner below is all that is needed
to start the bot from the project root.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) 
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src import main

if __name__ == "__main__":
    try:
        main.bot.run(main.load_config.DISCORD_TOKEN)
    except Exception:  
        import traceback
        traceback.print_exc()