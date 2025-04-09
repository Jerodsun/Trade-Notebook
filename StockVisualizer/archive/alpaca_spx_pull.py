import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from alpaca_trade_api.rest import TimeFrame

# Load environment variables
load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_API_SECRET")

BASE_URL = "https://paper-api.alpaca.markets"

# Initialize the Alpaca API
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# Get today's date
today = datetime.now().date()

# Get SPX data for the last weekday
symbol = "SPY"  # SPX is not directly available

# Calculate the last weekday
if today.weekday() == 0:  # Monday
    last_weekday = today - timedelta(days=3)
elif today.weekday() == 6:  # Sunday
    last_weekday = today - timedelta(days=2)
else:
    last_weekday = today - timedelta(days=1)

start_date = last_weekday.strftime("%Y-%m-%d")


end_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")

# Fetch data
bars = api.get_bars(symbol, TimeFrame.Minute, start="2025-03-20").df
