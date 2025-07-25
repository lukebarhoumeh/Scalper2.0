#!/usr/bin/env python3
"""
Fix Coinbase API Credentials
============================
Diagnose and fix API credential format issues
"""

import os
import re
from dotenv import load_dotenv

def fix_coinbase_credentials():
    """Fix the Coinbase API credential format"""
    print("🔧 Fixing Coinbase API Credentials...")
    
    load_dotenv()
    
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    
    print(f"Current API Key format: {api_key[:20]}..." if api_key else "No API key found")
    
    # The API key format looks like it includes the path, which is incorrect
    # Format should be just the UUID part
    if api_key and '/apiKeys/' in api_key:
        # Extract just the API key UUID
        parts = api_key.split('/apiKeys/')
        if len(parts) == 2:
            fixed_key = parts[1]  # Just the UUID part
            print(f"🔧 Fixed API Key: {fixed_key}")
            
            # Update .env file
            with open('.env', 'r') as f:
                content = f.read()
            
            content = content.replace(api_key, fixed_key)
            
            with open('.env', 'w') as f:
                f.write(content)
            
            print("✅ API key format fixed in .env file")
            return True
    
    print("⚠️  API key format appears to be incorrect")
    print("   Expected format: UUID (e.g., 96a83f0f-3c6b-43a2-a7be-70dd5a1940e5)")
    print("   Found format: Full path with '/apiKeys/' prefix")
    
    return False

def create_mock_trading_mode():
    """Create a mock trading mode that works without real API credentials"""
    print("\n🔧 Creating mock trading mode...")
    
    mock_config = """
# ─── MOCK TRADING MODE (No Real API Required) ─────────────────────────────────
# This mode simulates all API calls for testing and development
MOCK_TRADING_MODE=true
PAPER_TRADING=true

# Mock API credentials (not real, for simulation only)
COINBASE_API_KEY=mock_key_for_testing
COINBASE_API_SECRET=mock_secret_for_testing

# OpenAI still works for analysis
OPENAI_API_KEY=your_openai_api_key_here

# Performance optimizations
POLL_INTERVAL_SEC=60
REST_RATE_LIMIT_PER_S=5
USE_WS_FEED=false

# Mock market data
SIMULATE_MARKET_DATA=true
TRADE_COINS=BTC,ETH,SOL,DOGE,AVAX

# Reduced resource usage
TRADE_SIZE_USD=10.0
MAX_POSITION_USD=50.0
INVENTORY_CAP_USD=100.0
"""
    
    with open('.env.mock', 'w') as f:
        f.write(mock_config)
    
    print("✅ Mock trading mode created (.env.mock)")
    print("   Use this for testing without real API credentials")

def test_api_connection_methods():
    """Test different methods to connect to Coinbase API"""
    print("\n🔧 Testing different API connection methods...")
    
    from coinbase.rest import RESTClient
    import requests
    
    # Method 1: Try with original credentials
    load_dotenv()
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    
    methods = [
        ("Coinbase Advanced Trade API", "https://api.coinbase.com"),
        ("Coinbase Pro API (Legacy)", "https://api.pro.coinbase.com"),
        ("Coinbase Sandbox", "https://api.sandbox.coinbase.com")
    ]
    
    for name, base_url in methods:
        try:
            print(f"\n🧪 Testing {name}...")
            
            # Try a simple public endpoint first
            response = requests.get(f"{base_url}/products" if "sandbox" not in base_url else f"{base_url}/api/v3/brokerage/products", timeout=10)
            
            if response.status_code == 200:
                print(f"  ✅ Public API accessible")
                
                # Now try authenticated endpoint
                try:
                    client = RESTClient(
                        api_key=api_key,
                        api_secret=api_secret,
                        base_url=f"{base_url}/api/v3/brokerage/" if "v3" not in base_url else base_url
                    )
                    accounts = client.get_accounts()
                    print(f"  ✅ {name} authentication successful")
                    return True
                except Exception as e:
                    print(f"  ❌ {name} authentication failed: {e}")
            else:
                print(f"  ❌ {name} not accessible: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ {name} error: {e}")
    
    return False

def main():
    """Fix API credential issues"""
    print("🚀 Coinbase API Credential Fixer")
    print("=" * 40)
    
    # Try to fix credentials
    if fix_coinbase_credentials():
        print("\n🧪 Testing fixed credentials...")
        if test_api_connection_methods():
            print("\n✅ API credentials fixed and working!")
        else:
            print("\n⚠️  Credentials fixed but still not working")
            print("   This may be due to:")
            print("   - Expired API keys")
            print("   - Insufficient permissions")
            print("   - Wrong environment (sandbox vs production)")
            create_mock_trading_mode()
    else:
        print("\n⚠️  Could not automatically fix credentials")
        create_mock_trading_mode()
    
    print("\n📋 Manual Steps to Fix Coinbase API:")
    print("1. Go to https://www.coinbase.com/settings/api")
    print("2. Create new API keys for 'Advanced Trade'")
    print("3. Grant permissions: 'trade', 'view', 'transfer'")
    print("4. Use the KEY (not the full path) in COINBASE_API_KEY")
    print("5. Use the SECRET in COINBASE_API_SECRET")
    
    print("\n🎯 For now, you can test with mock mode:")
    print("   cp .env.mock .env")
    print("   python3 run_optimized_bot.py")

if __name__ == "__main__":
    main()