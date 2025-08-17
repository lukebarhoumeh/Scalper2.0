# coinbase_api.py
"""
Basic coinbase client interface → good for manual testing.
"""

from coinbase.rest import RESTClient
from config import COINBASE_API_KEY, COINBASE_API_SECRET

client = RESTClient(api_key=COINBASE_API_KEY, api_secret=COINBASE_API_SECRET)

if __name__ == "__main__":
    # Experiment freely: list accounts, get prices, etc.
    print("CB client ready:")
    
    # Get accounts with proper null check
    accounts_response = client.get_accounts()
    if accounts_response and accounts_response.accounts:
        accounts = accounts_response.accounts
        print(f"  {len(accounts)} accounts found")
        for acc in accounts:
            print(f"    {acc.currency}: {acc.available_balance}")
    else:
        print("  No accounts found")
