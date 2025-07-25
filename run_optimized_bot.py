#!/usr/bin/env python3
"""
Optimized ScalperBot Launcher
High-performance version with connection pooling and caching
"""

import os
import sys
import signal
import time
import logging
from datetime import datetime
import asyncio

# Performance optimizations
os.environ['PYTHONOPTIMIZE'] = '1'  # Enable Python optimizations
os.environ['PYTHONUNBUFFERED'] = '1'  # Improve logging performance

def setup_optimized_environment():
    """Setup optimized environment"""
    # Use optimized config if available
    if os.path.exists('.env.optimized'):
        # Copy optimized config to main .env
        with open('.env.optimized', 'r') as f:
            optimized = f.read()
        with open('.env', 'w') as f:
            f.write(optimized)
        print("📊 Using optimized configuration")
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Set memory optimizations
    import gc
    gc.set_threshold(700, 10, 10)  # More aggressive garbage collection

def main():
    """Launch optimized bot"""
    print("🚀 Starting Optimized ScalperBot")
    print("=" * 40)
    
    setup_optimized_environment()
    
    try:
        # Import bot after environment setup
        import unified_scalperbot_v2
        
        print(f"📝 Paper Trading: {unified_scalperbot_v2.PAPER_TRADING}")
        print(f"💱 Trading Pairs: {len(unified_scalperbot_v2.TRADE_COINS)}")
        
        # Initialize bot
        bot = unified_scalperbot_v2.UnifiedScalperBot()
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print("\n🛑 Shutdown signal received")
            bot.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start bot
        print("🎯 Starting trading loop...")
        bot.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("📊 Bot stopped")

if __name__ == "__main__":
    main()
