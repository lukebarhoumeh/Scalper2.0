#!/usr/bin/env python3
"""
Analyze trading performance from trades.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys

def analyze_trades(csv_file='logs/trades.csv', start_line=8256):
    """Analyze trading performance starting from specified line."""
    
    # Read the CSV file
    try:
        # Read all trades
        df = pd.read_csv(csv_file, header=None, names=[
            'order_id', 'timestamp', 'product_id', 'side', 
            'qty_base', 'price', 'size_usd', 'strategy', 'pnl'
        ])
        
        # Filter from start line
        df_recent = df.iloc[start_line-1:].copy()
        
        # Convert timestamp to datetime
        df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp'])
        
        # Clean up product_id (remove extra spaces)
        df_recent['product_id'] = df_recent['product_id'].str.strip()
        
        # Convert numeric columns
        df_recent['qty_base'] = pd.to_numeric(df_recent['qty_base'])
        df_recent['price'] = pd.to_numeric(df_recent['price'])
        df_recent['size_usd'] = pd.to_numeric(df_recent['size_usd'])
        df_recent['pnl'] = pd.to_numeric(df_recent['pnl'])
        
        print(f"\n{'='*60}")
        print("TRADE ANALYSIS REPORT")
        print(f"{'='*60}\n")
        
        # Basic statistics
        print(f"Total trades analyzed: {len(df_recent)}")
        print(f"Date range: {df_recent['timestamp'].min()} to {df_recent['timestamp'].max()}")
        print(f"Time span: {(df_recent['timestamp'].max() - df_recent['timestamp'].min()).total_seconds() / 3600:.1f} hours")
        
        # Trades by product
        print(f"\n{'─'*40}")
        print("TRADES BY PRODUCT:")
        print(f"{'─'*40}")
        product_stats = df_recent.groupby('product_id').agg({
            'order_id': 'count',
            'size_usd': ['sum', 'mean']
        }).round(2)
        product_stats.columns = ['Count', 'Total Volume ($)', 'Avg Size ($)']
        print(product_stats)
        
        # Trades by side
        print(f"\n{'─'*40}")
        print("TRADES BY SIDE:")
        print(f"{'─'*40}")
        side_stats = df_recent.groupby('side').agg({
            'order_id': 'count',
            'size_usd': 'sum'
        }).round(2)
        side_stats.columns = ['Count', 'Total Volume ($)']
        print(side_stats)
        
        # Buy/Sell ratio by product
        print(f"\n{'─'*40}")
        print("BUY/SELL RATIO BY PRODUCT:")
        print(f"{'─'*40}")
        for product in df_recent['product_id'].unique():
            product_trades = df_recent[df_recent['product_id'] == product]
            buys = len(product_trades[product_trades['side'] == 'buy'])
            sells = len(product_trades[product_trades['side'] == 'sell'])
            total = buys + sells
            if total > 0:
                buy_pct = (buys / total) * 100
                print(f"{product}: {buys} buys ({buy_pct:.1f}%), {sells} sells ({100-buy_pct:.1f}%)")
        
        # Trade frequency analysis
        print(f"\n{'─'*40}")
        print("TRADE FREQUENCY ANALYSIS:")
        print(f"{'─'*40}")
        
        # Calculate time between trades
        df_recent['time_diff'] = df_recent['timestamp'].diff()
        avg_time_between_trades = df_recent['time_diff'].mean()
        
        print(f"Average time between trades: {avg_time_between_trades.total_seconds() / 60:.1f} minutes")
        print(f"Min time between trades: {df_recent['time_diff'].min().total_seconds():.1f} seconds")
        print(f"Max time between trades: {df_recent['time_diff'].max().total_seconds() / 60:.1f} minutes")
        
        # Trade clustering analysis
        df_recent['minute'] = df_recent['timestamp'].dt.floor('T')
        trades_per_minute = df_recent.groupby('minute').size()
        
        print(f"\nTrades per minute stats:")
        print(f"  Average: {trades_per_minute.mean():.2f}")
        print(f"  Max: {trades_per_minute.max()}")
        print(f"  Std Dev: {trades_per_minute.std():.2f}")
        
        # Price movement analysis
        print(f"\n{'─'*40}")
        print("PRICE MOVEMENT ANALYSIS:")
        print(f"{'─'*40}")
        
        for product in df_recent['product_id'].unique():
            product_trades = df_recent[df_recent['product_id'] == product].copy()
            if len(product_trades) > 1:
                product_trades['price_change_pct'] = product_trades['price'].pct_change() * 100
                avg_move = product_trades['price_change_pct'].abs().mean()
                max_move = product_trades['price_change_pct'].abs().max()
                
                print(f"\n{product}:")
                print(f"  Average price movement: {avg_move:.3f}%")
                print(f"  Max price movement: {max_move:.3f}%")
                print(f"  Price range: ${product_trades['price'].min():.4f} - ${product_trades['price'].max():.4f}")
        
        # Trade size analysis
        print(f"\n{'─'*40}")
        print("TRADE SIZE ANALYSIS:")
        print(f"{'─'*40}")
        
        print(f"Average trade size: ${df_recent['size_usd'].mean():.2f}")
        print(f"Min trade size: ${df_recent['size_usd'].min():.2f}")
        print(f"Max trade size: ${df_recent['size_usd'].max():.2f}")
        print(f"Std Dev: ${df_recent['size_usd'].std():.2f}")
        
        # Risk exposure analysis
        print(f"\n{'─'*40}")
        print("RISK EXPOSURE ANALYSIS:")
        print(f"{'─'*40}")
        
        # Calculate net position for each product
        for product in df_recent['product_id'].unique():
            product_trades = df_recent[df_recent['product_id'] == product]
            
            # Calculate net position
            buy_qty = product_trades[product_trades['side'] == 'buy']['qty_base'].sum()
            sell_qty = product_trades[product_trades['side'] == 'sell']['qty_base'].sum()
            net_position = buy_qty - sell_qty
            
            # Get last price
            last_price = product_trades['price'].iloc[-1]
            net_usd = net_position * last_price
            
            print(f"\n{product}:")
            print(f"  Net position: {net_position:.4f} units (${net_usd:.2f})")
            print(f"  Total buys: {buy_qty:.4f} units")
            print(f"  Total sells: {sell_qty:.4f} units")
        
        # Pattern analysis
        print(f"\n{'─'*40}")
        print("PATTERN ANALYSIS:")
        print(f"{'─'*40}")
        
        # Check for consecutive trades in same direction
        for product in df_recent['product_id'].unique():
            product_trades = df_recent[df_recent['product_id'] == product].copy()
            
            if len(product_trades) > 1:
                # Count consecutive trades
                product_trades['side_change'] = product_trades['side'] != product_trades['side'].shift()
                trade_runs = product_trades['side_change'].cumsum()
                run_lengths = product_trades.groupby(trade_runs).size()
                
                max_consecutive = run_lengths.max()
                avg_consecutive = run_lengths.mean()
                
                print(f"\n{product}:")
                print(f"  Max consecutive trades (same side): {max_consecutive}")
                print(f"  Average run length: {avg_consecutive:.1f}")
        
        return df_recent
        
    except Exception as e:
        print(f"Error analyzing trades: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run analysis
    df = analyze_trades()
    
    if df is not None:
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS:")
        print(f"{'='*60}\n")
        
        print("1. TRADE FREQUENCY: The bot is trading very frequently (often < 1 minute between trades)")
        print("   → Consider increasing cooldown period to reduce overtrading")
        print("\n2. POSITION IMBALANCE: Check for significant net positions that indicate directional bias")
        print("   → May need better mean reversion signals or position limits")
        print("\n3. TRADE CLUSTERING: Multiple trades happening in quick succession")
        print("   → Implement better signal filtering to avoid redundant trades")
        print("\n4. SMALL PRICE MOVEMENTS: Trading on very small price changes")
        print("   → Consider increasing spread threshold for better profit potential") 