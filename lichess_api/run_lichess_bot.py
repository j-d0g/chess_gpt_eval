#!/usr/bin/env python3
"""
Simple launcher for ChessGPT Lichess Bot
"""

import os
import sys

def main():
    print("🤖 ChessGPT Lichess Bot Launcher")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("chess_gpt_eval"):
        print("❌ Error: chess_gpt_eval directory not found")
        print("   Please run this script from the ChessGPT root directory")
        return
    
    # Check if gpt_inputs/prompt.txt exists
    if not os.path.exists("gpt_inputs/prompt.txt"):
        print("⚠️  Warning: gpt_inputs/prompt.txt not found")
        print("   The bot will work but may not have the optimal prompt")
    
    # Import and run the bot
    try:
        from lichess_bot import main as bot_main
        bot_main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you've installed the requirements: pip install -r requirements_bot.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 