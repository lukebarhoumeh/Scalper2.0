# coinbase_api.py
from coinbase.rest import RESTClient
from config import COINBASE_API_KEY, COINBASE_API_SECRET

def get_coinbase_client():
    return RESTClient(
        api_key=COINBASE_API_KEY,
        api_secret=COINBASE_API_SECRET,
    )

def test_coinbase_api():
    client = get_coinbase_client()
    resp = client.get_accounts()
    accounts = resp.accounts          # list of dicts
    print(f"Found {len(accounts)} accounts")
    for acct in accounts:
        bal = acct["available_balance"]       # dict with keys 'value' and 'currency'
        print(f"{acct['currency']}: {bal['value']} {bal['currency']}")

if __name__ == "__main__":
    test_coinbase_api()
