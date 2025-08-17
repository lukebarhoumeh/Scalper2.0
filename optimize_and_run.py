#!/usr/bin/env python3
"""
Optimize and Run Trading Bot

This script configures the bot with optimal settings and launches it.
Usage: python optimize_and_run.py [mode]
Modes: aggressive, balanced, conservative, synthetic
"""

import sys
import os
import asyncio
from pathlib import Path

# Add bot directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bot.config_optimizer import ConfigOptimizer, create_optimal_env_file
from bot.app import App
from bot.config import load_config


def print_banner(mode: str):
    """Print startup banner"""
    banner = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                   SCALPER BOT 2.0 - PROFIT ENGINE            ║
    ║                                                              ║
    ║  Mode: {mode.upper():^20}                         ║
    ║  Target: Maximum Profitability                               ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def setup_optimal_config(mode: str):
    """Setup optimal configuration based on mode"""
    
    # Get configuration
    if mode == "aggressive":
        config = ConfigOptimizer.get_aggressive_config()
        print("🚀 Loading AGGRESSIVE configuration - Maximum trade frequency")
    elif mode == "balanced":
        config = ConfigOptimizer.get_balanced_config()
        print("⚖️  Loading BALANCED configuration - Steady profits")
    elif mode == "conservative":
        config = ConfigOptimizer.get_conservative_config()
        print("🛡️  Loading CONSERVATIVE configuration - Risk protection")
    elif mode == "synthetic":
        config = ConfigOptimizer.get_synthetic_optimized_config()
        print("🧪 Loading SYNTHETIC configuration - Optimized for testing")
    else:
        print(f"⚠️  Unknown mode '{mode}', using balanced configuration")
        config = ConfigOptimizer.get_balanced_config()
        mode = "balanced"
    
    # Apply configuration to environment
    ConfigOptimizer.apply_config(config)
    
    # Save optimal .env file for reference
    env_content = create_optimal_env_file(mode)
    env_path = Path(f".env.{mode}")
    with open(env_path, "w") as f:
        f.write(env_content)
    print(f"💾 Saved configuration to {env_path}")
    
    return config


def print_config_summary(config: dict):
    """Print configuration summary"""
    print("\n📊 Configuration Summary:")
    print(f"  • RSI Thresholds: Buy={config.get('RSI_BUY_THRESHOLD', 'N/A')} / Sell={config.get('RSI_SELL_THRESHOLD', 'N/A')}")
    print(f"  • SMA Periods: Fast={config.get('SMA_FAST', 'N/A')} / Slow={config.get('SMA_SLOW', 'N/A')}")
    print(f"  • Max Spread: {config.get('MAX_SPREAD_BPS', 'N/A')} bps")
    print(f"  • Exits: TP={config.get('TAKE_PROFIT_BPS', 'N/A')}bps / SL={config.get('STOP_LOSS_BPS', 'N/A')}bps / Trail={config.get('TRAILING_STOP_BPS', 'N/A')}bps")
    print(f"  • Risk: {float(config.get('RISK_PERCENT', '0.02')) * 100:.1f}% per trade")
    print(f"  • Activity Target: {config.get('ACTIVITY_TARGET_TPH', 'N/A')} trades/hour")
    print(f"  • Poll Interval: {config.get('POLL_INTERVAL_SEC', 'N/A')}s")
    print()


async def run_bot():
    """Run the trading bot"""
    try:
        # Load configuration
        cfg = load_config()
        
        # Create and run app
        app = App(cfg)
        
        print("🤖 Bot initialized successfully")
        print("💰 Starting capital: $" + f"{cfg.starting_cash_usd:,.2f}")
        print("📈 Trading pairs: " + ", ".join(cfg.trading_pairs))
        print("\n✅ Bot is running! Press Ctrl+C to stop.\n")
        
        # Run the bot
        await app.run()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested...")
        app.request_shutdown()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


def main():
    """Main entry point"""
    # Get mode from command line or default to synthetic
    mode = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    
    # Print banner
    print_banner(mode)
    
    # Setup configuration
    config = setup_optimal_config(mode)
    
    # Print configuration summary
    print_config_summary(config)
    
    # Special message for synthetic mode
    if mode == "synthetic":
        print("💡 TIP: Running in synthetic mode with optimized parameters.")
        print("   Trades should start appearing within 30-60 seconds.")
        print("   If no trades appear, the bot will auto-adjust parameters.\n")
    
    # Run the bot
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped. Check logs/ledger.csv for trade history.")
        print("💰 Check logs/profit_history.json for withdrawal history.")


if __name__ == "__main__":
    main()
