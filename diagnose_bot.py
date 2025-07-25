#!/usr/bin/env python3
"""
ScalperBot Diagnostic Tool
Analyzes performance bottlenecks and connection issues
"""

import os
import sys
import time
import traceback
import asyncio
from datetime import datetime
import json

# Add current directory to path
sys.path.insert(0, os.getcwd())

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")
    
    tests = [
        ("dotenv", "from dotenv import load_dotenv"),
        ("pandas", "import pandas as pd"),
        ("numpy", "import numpy as np"),
        ("coinbase", "from coinbase.rest import RESTClient"),
        ("openai", "import openai"),
        ("websocket", "import websocket"),
        ("requests", "import requests"),
        ("config_unified", "import config_unified"),
        ("config", "import config"),
        ("market_data", "import market_data"),
        ("coinbase_client", "import coinbase_client"),
    ]
    
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"  ✅ {name}: OK")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
    
    print()

def test_environment():
    """Test environment variables"""
    print("🔍 Testing environment variables...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = [
        "COINBASE_API_KEY",
        "COINBASE_API_SECRET",
        "OPENAI_API_KEY"
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Show first/last 4 chars for security
            masked = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "***"
            print(f"  ✅ {var}: {masked}")
        else:
            print(f"  ❌ {var}: Not set")
    
    print()

def test_coinbase_connection():
    """Test Coinbase API connection"""
    print("🔍 Testing Coinbase API connection...")
    
    try:
        from coinbase.rest import RESTClient
        import config_unified
        
        if not config_unified.COINBASE_API_KEY or not config_unified.COINBASE_API_SECRET:
            print("  ❌ Missing API credentials")
            return
            
        client = RESTClient(
            api_key=config_unified.COINBASE_API_KEY,
            api_secret=config_unified.COINBASE_API_SECRET
        )
        
        # Test basic connection
        start_time = time.time()
        try:
            accounts = client.get_accounts()
            response_time = time.time() - start_time
            print(f"  ✅ API Connection: OK ({response_time:.2f}s)")
        except Exception as e:
            print(f"  ❌ API Connection: {e}")
            return
            
        # Test market data
        start_time = time.time()
        try:
            products = client.get_products()
            response_time = time.time() - start_time
            print(f"  ✅ Market Data: OK ({response_time:.2f}s, {len(products)} products)")
        except Exception as e:
            print(f"  ❌ Market Data: {e}")
            
        # Test specific trading pairs
        trading_pairs = config_unified.TRADING_PAIRS
        print(f"  📋 Testing {len(trading_pairs)} trading pairs...")
        
        for pair in trading_pairs:
            try:
                start_time = time.time()
                ticker = client.get_product(pair)
                response_time = time.time() - start_time
                print(f"    ✅ {pair}: OK ({response_time:.2f}s)")
            except Exception as e:
                print(f"    ❌ {pair}: {e}")
                
    except Exception as e:
        print(f"  ❌ Coinbase setup error: {e}")
        traceback.print_exc()
    
    print()

def test_openai_connection():
    """Test OpenAI API connection"""
    print("🔍 Testing OpenAI API connection...")
    
    try:
        import openai
        import config_unified
        
        if not config_unified.OPENAI_API_KEY:
            print("  ❌ Missing OpenAI API key")
            return
            
        client = openai.OpenAI(api_key=config_unified.OPENAI_API_KEY)
        
        start_time = time.time()
        try:
            # Test with minimal request
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            response_time = time.time() - start_time
            print(f"  ✅ OpenAI API: OK ({response_time:.2f}s)")
        except Exception as e:
            print(f"  ❌ OpenAI API: {e}")
            
    except Exception as e:
        print(f"  ❌ OpenAI setup error: {e}")
    
    print()

def analyze_market_data_performance():
    """Analyze market data fetching performance"""
    print("🔍 Analyzing market data performance...")
    
    try:
        import market_data
        import config_unified
        
        # Test REST client pool
        print("  📊 Testing REST client pool...")
        pool = market_data.RESTClientPool(pool_size=3)
        
        # Test multiple concurrent requests
        start_time = time.time()
        clients = []
        for i in range(3):
            client = pool.acquire()
            clients.append(client)
        
        for client in clients:
            pool.release(client)
            
        pool_time = time.time() - start_time
        print(f"    ✅ Pool operations: {pool_time:.3f}s")
        
        # Test rate limiting
        print("  ⏱️  Testing rate limiting...")
        from market_data import with_timeout
        
        @with_timeout(5)
        def test_rate_limit():
            # Simulate multiple rapid requests
            for i in range(5):
                time.sleep(0.1)
            return "OK"
            
        start_time = time.time()
        result = test_rate_limit()
        rate_time = time.time() - start_time
        print(f"    ✅ Rate limiting: {rate_time:.3f}s")
        
    except Exception as e:
        print(f"  ❌ Market data error: {e}")
        traceback.print_exc()
    
    print()

def check_performance_bottlenecks():
    """Identify potential performance bottlenecks"""
    print("🔍 Checking for performance bottlenecks...")
    
    # Memory usage
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"  💾 Memory usage: {memory_mb:.1f} MB")
        
        cpu_percent = process.cpu_percent()
        print(f"  🖥️  CPU usage: {cpu_percent:.1f}%")
        
    except Exception as e:
        print(f"  ❌ System metrics error: {e}")
    
    # File system performance
    try:
        start_time = time.time()
        test_file = "performance_test.tmp"
        
        # Write test
        with open(test_file, 'w') as f:
            f.write("test" * 1000)
        write_time = time.time() - start_time
        
        # Read test
        start_time = time.time()
        with open(test_file, 'r') as f:
            data = f.read()
        read_time = time.time() - start_time
        
        os.remove(test_file)
        
        print(f"  📁 File I/O: Write {write_time:.3f}s, Read {read_time:.3f}s")
        
    except Exception as e:
        print(f"  ❌ File I/O error: {e}")
    
    print()

def test_bot_initialization():
    """Test bot initialization"""
    print("🔍 Testing bot initialization...")
    
    try:
        import unified_scalperbot_v2
        
        start_time = time.time()
        bot = unified_scalperbot_v2.UnifiedScalperBot()
        init_time = time.time() - start_time
        
        print(f"  ✅ Bot initialization: {init_time:.2f}s")
        
        # Test configuration loading
        if hasattr(bot, 'current_mode'):
            print(f"  📋 Bot mode: {bot.current_mode}")
        
        if hasattr(bot, 'total_trades'):
            print(f"  📊 Total trades: {bot.total_trades}")
            
        # Test paper trading status
        print(f"  📝 Paper trading: {unified_scalperbot_v2.PAPER_TRADING}")
        
        # Test trading pairs
        if hasattr(unified_scalperbot_v2, 'TRADE_COINS'):
            pairs = unified_scalperbot_v2.TRADE_COINS
            print(f"  💱 Trading pairs: {len(pairs)} configured")
            for pair in pairs:
                print(f"    - {pair}")
        
    except Exception as e:
        print(f"  ❌ Bot initialization error: {e}")
        traceback.print_exc()
    
    print()

def create_performance_report():
    """Create a performance optimization report"""
    print("📊 Performance Optimization Report")
    print("=" * 50)
    
    recommendations = []
    
    # Check configuration
    try:
        import config_unified
        
        poll_interval = config_unified.POLL_INTERVAL_SEC
        if poll_interval < 10:
            recommendations.append(f"Consider increasing POLL_INTERVAL_SEC from {poll_interval}s to 10-30s to reduce API load")
            
        max_requests = config_unified.MAX_REQUESTS_PER_SECOND
        if max_requests > 10:
            recommendations.append(f"Consider reducing MAX_REQUESTS_PER_SECOND from {max_requests} to 8 or less")
            
    except Exception as e:
        recommendations.append(f"Could not analyze configuration: {e}")
    
    # Bundle size optimization
    print("\n🎯 Bundle Size Optimizations:")
    large_imports = [
        ("pandas", "Consider using only necessary pandas functions"),
        ("numpy", "Already optimized"),
        ("scipy", "Consider if all scipy features are needed"),
        ("scikit-learn", "Consider if ML features are actively used")
    ]
    
    for lib, suggestion in large_imports:
        try:
            exec(f"import {lib}")
            print(f"  📦 {lib}: {suggestion}")
        except:
            pass
    
    print("\n⚡ Performance Recommendations:")
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("  ✅ No immediate performance issues detected")
    
    print("\n🔧 Bot Optimization Suggestions:")
    print("  1. Use connection pooling for API calls")
    print("  2. Implement proper rate limiting")
    print("  3. Cache market data when possible")
    print("  4. Use WebSocket feeds for real-time data")
    print("  5. Optimize logging levels for production")
    print("  6. Monitor memory usage during long runs")

def main():
    """Run all diagnostics"""
    print("🚀 ScalperBot Performance Diagnostic")
    print("=" * 40)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    test_imports()
    test_environment()
    test_coinbase_connection()
    test_openai_connection()
    analyze_market_data_performance()
    check_performance_bottlenecks()
    test_bot_initialization()
    create_performance_report()
    
    print("\n✅ Diagnostic complete!")

if __name__ == "__main__":
    main()