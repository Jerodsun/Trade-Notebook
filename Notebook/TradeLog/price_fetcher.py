import requests
import json
from datetime import datetime

def get_index_prices():
    """
    Fetches delayed prices for SPX and NDX from Yahoo Finance.
    Returns a dictionary with current prices.
    """
    symbols = ["^SPX", "^NDX"]
    results = {}
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for symbol in symbols:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1m&range=1d"
            response = requests.get(url, headers=headers, timeout=5)
            data = response.json()
            
            # Extract the latest regular market price
            meta = data['chart']['result'][0]['meta']
            current_price = meta.get('regularMarketPrice')
            prev_close = meta.get('previousClose')
            
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            
            results[symbol] = {
                'price': current_price,
                'change': change,
                'percent': change_percent
            }
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            results[symbol] = None
            
    return results

def get_index_history(symbol, timeframe="1d"):
    """
    Fetches historical data for a symbol.
    timeframe: "1d" (today) or "5d" (this week)
    """
    interval = "2m" if timeframe == "1d" else "15m"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={timeframe}"
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        prices = result['indicators']['quote'][0]['close']
        
        # Filter out None values in prices
        valid_data = [(datetime.fromtimestamp(ts), pr) for ts, pr in zip(timestamps, prices) if pr is not None]
        if not valid_data:
            return None
            
        times, filtered_prices = zip(*valid_data)
        return {
            'times': times,
            'prices': filtered_prices
        }
    except Exception as e:
        print(f"Error fetching history for {symbol}: {e}")
        return None
