#!/usr/bin/env python3
"""Test Coinbase API credentials with different approaches"""
import os
from dotenv import load_dotenv
load_dotenv()

def test_coinbase_credentials():
    api_key = os.getenv('COINBASE_API_KEY')
    api_secret = os.getenv('COINBASE_API_SECRET')
    
    print(f"API Key format: {api_key[:20]}...{api_key[-10:] if len(api_key) > 30 else api_key}")
    print(f"Secret format: {'EC PRIVATE KEY' if 'EC PRIVATE KEY' in api_secret else 'Unknown format'}")
    
    # Test different client configurations
    from coinbase.rest import RESTClient
    
    try:
        # Try with sandbox first
        client = RESTClient(
            api_key=api_key,
            api_secret=api_secret,
            base_url="https://api.coinbase.com/api/v3/brokerage/"  # Production URL
        )
        accounts = client.get_accounts()
        print("✅ Production API works")
        return True
    except Exception as e:
        print(f"❌ Production API failed: {e}")
        
    try:
        # Try sandbox
        client = RESTClient(
            api_key=api_key,
            api_secret=api_secret,
            base_url="https://api.sandbox.coinbase.com/api/v3/brokerage/"
        )
        accounts = client.get_accounts()
        print("✅ Sandbox API works")
        return True
    except Exception as e:
        print(f"❌ Sandbox API failed: {e}")
    
    return False

if __name__ == "__main__":
    test_coinbase_credentials()
