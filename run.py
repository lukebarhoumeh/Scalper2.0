#!/usr/bin/env python3
"""
ScalperBot 2.0 - Unified Entry Point
===================================
Single entry script for running the production trading bot
"""

import os
import sys
import gc
import logging
from colorama import Fore, Style

# Set optimization flags
os.environ["PYTHONOPTIMIZE"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

# Configure garbage collection for better performance
gc.set_threshold(700, 10, 10)

# Import the unified bot
from unified_scalperbot_v2 import UnifiedScalperBot
from config_unified import validate_config, PAPER_TRADING, TRADING_PAIRS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for ScalperBot 2.0"""
    
    print(f"{Fore.CYAN}")
    print("=" * 60)
    print("        ScalperBot 2.0 - Production Trading System")
    print("=" * 60)
    print(f"{Style.RESET_ALL}")
    
    # Validate configuration
    errors = validate_config()
    if errors:
        logger.error(f"{Fore.RED}Configuration errors found:{Style.RESET_ALL}")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    # Display trading mode
    mode = "PAPER TRADING" if PAPER_TRADING else "LIVE TRADING"
    mode_color = Fore.YELLOW if PAPER_TRADING else Fore.RED
    
    print(f"\n{mode_color}Mode: {mode}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Trading Pairs: {', '.join(TRADING_PAIRS)}{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Starting bot...{Style.RESET_ALL}\n")
    
    try:
        # Create and run the bot
        bot = UnifiedScalperBot()
        bot.run()
        
    except KeyboardInterrupt:
        logger.info(f"{Fore.YELLOW}Shutting down gracefully...{Style.RESET_ALL}")
        
    except Exception as e:
        logger.error(f"{Fore.RED}Fatal error: {e}{Style.RESET_ALL}", exc_info=True)
        sys.exit(1)
        
    finally:
        print(f"\n{Fore.CYAN}Bot stopped. Thank you for using ScalperBot 2.0!{Style.RESET_ALL}")


if __name__ == "__main__":
    main() 