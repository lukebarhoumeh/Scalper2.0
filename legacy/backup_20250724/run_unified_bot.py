#!/usr/bin/env python3
"""
Simple launcher for Unified ScalperBot
"""

import os
import sys
import asyncio

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the bot
from unified_scalperbot import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 