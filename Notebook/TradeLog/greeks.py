import numpy as np
from scipy.stats import norm
from datetime import datetime

def black_scholes_price(S, K, T, r, sigma, option_type='Call'):
    """
    S: current price of the underlying asset
    K: strike price
    T: time to expiration (in years)
    r: risk-free interest rate (e.g., 0.05 for 5%)
    sigma: volatility of the underlying asset
    option_type: 'Call' or 'Put'
    """
    if T <= 0:
        if option_type == 'Call':
            return max(0, S - K)
        else:
            return max(0, K - S)
            
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'Call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    return price

def calculate_iv(S, K, T, r, market_price, option_type='Call'):
    """
    Calculates Implied Volatility using Newton-Raphson method.
    """
    if T <= 0:
        return 0.0
        
    # Initial guess
    sigma = 0.5
    for i in range(100):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        
        # Vega calculation for Newton-Raphson
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        diff = market_price - price
        
        if abs(diff) < 1e-6:
            return sigma
            
        if vega > 0:
            sigma = sigma + diff / vega
        else:
            return 0.0 # Could not converge
            
        if sigma <= 0:
            sigma = 0.001 # Keep sigma positive
            
    return sigma

def calculate_greeks(S, K, T, r, sigma, option_type='Call'):
    """
    Returns a dictionary of Delta, Gamma, Theta, Vega.
    """
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == 'Call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
        
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100 # per 1% change in IV
    
    # Theta
    if option_type == 'Call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
    return {
        'delta': float(delta),
        'gamma': float(gamma),
        'theta': float(theta),
        'vega': float(vega)
    }

def get_days_to_expiration(expiration_date_str):
    """
    expiration_date_str: 'YYYY-MM-DD'
    Returns time to expiration in years.
    """
    expiry = datetime.strptime(expiration_date_str, '%Y-%m-%d')
    now = datetime.now()
    delta = expiry - now
    days = delta.days + (delta.seconds / 86400)
    return max(0, days) / 365.0
