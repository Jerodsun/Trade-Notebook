import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from tda import auth, client
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Get API credentials from environment variables
API_KEY = os.getenv("TDA_API_KEY")
REDIRECT_URI = os.getenv("TDA_REDIRECT_URI")
TOKEN_PATH = os.getenv("TDA_TOKEN_PATH", "token.json")

# Connect to TD Ameritrade
try:
    c = auth.client_from_token_file(TOKEN_PATH, API_KEY)
except FileNotFoundError:
    from selenium import webdriver

    with webdriver.Chrome() as driver:
        c = auth.client_from_login_flow(driver, API_KEY, REDIRECT_URI, TOKEN_PATH)


class SPXOptionIntraminuteTrader:
    def __init__(self):
        self.client = c
        self.symbol = "$SPX.X"
        self.position_open = False
        self.entry_price = 0
        self.entry_time = None
        self.position_type = None  # 'call' or 'put'
        self.stop_loss_pct = 0.15  # 15% stop loss
        self.profit_target_pct = 0.30  # 30% profit target
        self.spike_threshold = 0.25  # Minimum % change to qualify as a spike
        self.lookback_period = 5  # Minutes to look back for spike detection

    def get_current_price(self):
        """Get current SPX price"""
        response = self.client.get_quote(self.symbol)
        data = response.json()
        return data[self.symbol]["lastPrice"]

    def get_minute_data(self):
        """Get 1-minute price data for the last 15 minutes"""
        end_date = dt.datetime.now()
        start_date = end_date - dt.timedelta(minutes=15)

        response = self.client.get_price_history(
            symbol=self.symbol,
            period_type=client.Client.PriceHistory.PeriodType.DAY,
            period=client.Client.PriceHistory.Period.ONE_DAY,
            frequency_type=client.Client.PriceHistory.FrequencyType.MINUTE,
            frequency=client.Client.PriceHistory.Frequency.EVERY_MINUTE,
            start_datetime=start_date,
            end_datetime=end_date,
        )

        data = response.json()
        df = pd.DataFrame(data["candles"])
        df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
        return df

    def detect_spike(self, df):
        """Detect if there's a significant price spike in the last few minutes"""
        if len(df) < self.lookback_period + 1:
            return False, None

        # Calculate minute-to-minute percentage changes
        df["pct_change"] = df["close"].pct_change() * 100

        # Get the most recent change
        latest_change = df["pct_change"].iloc[-1]

        if abs(latest_change) > self.spike_threshold:
            # Spike detected
            direction = "up" if latest_change > 0 else "down"
            return True, direction
        return False, None

    def find_nearest_strike(self, current_price):
        """Find the nearest option strike price"""
        # SPX options typically have strikes in increments of 5
        return round(current_price / 5) * 5

    def get_option_chain(self, strike_price):
        """Get option chain for the nearest expiration and strike"""
        today = dt.date.today()

        # Find the next expiration (typically same day for SPX 0DTE options)
        response = self.client.get_option_chain(
            symbol=self.symbol,
            contract_type=client.Client.Options.ContractType.ALL,
            strike=strike_price,
            from_date=today,
            to_date=today + dt.timedelta(days=1),
        )

        chain = response.json()
        return chain

    def select_option(self, chain, direction, strike_price):
        """Select appropriate option based on spike direction"""
        # Use calls for upward spikes, puts for downward spikes
        option_type = "calls" if direction == "up" else "puts"

        # Find the option with strike_price that expires today
        expirations = (
            chain["callExpDateMap"]
            if option_type == "calls"
            else chain["putExpDateMap"]
        )

        # Get the first expiration (nearest)
        first_expiry = list(expirations.keys())[0]
        strikes = expirations[first_expiry]

        # Find our target strike
        strike_key = str(strike_price)
        if strike_key in strikes:
            return strikes[strike_key][0], option_type
        else:
            # Find closest available strike
            available_strikes = [float(s) for s in strikes.keys()]
            closest_strike = str(
                min(available_strikes, key=lambda x: abs(x - strike_price))
            )
            return strikes[closest_strike][0], option_type

    def execute_trade(self, option, option_type):
        """Execute the option trade"""
        # In a real implementation, you would place an actual order here
        # This is just a simulation
        self.position_open = True
        self.entry_price = option["last"]
        self.entry_time = dt.datetime.now()
        self.position_type = option_type

        print(
            f"Executed {option_type} trade at {self.entry_price} at {self.entry_time}"
        )
        print(
            f"Option details: Strike={option['strikePrice']}, Expiration={option['expirationDate']}"
        )

        # Set stop loss and profit target prices
        self.stop_loss = (
            self.entry_price * (1 - self.stop_loss_pct)
            if option_type == "calls"
            else self.entry_price * (1 + self.stop_loss_pct)
        )
        self.profit_target = (
            self.entry_price * (1 + self.profit_target_pct)
            if option_type == "calls"
            else self.entry_price * (1 - self.profit_target_pct)
        )

        print(f"Stop loss set at: {self.stop_loss}")
        print(f"Profit target set at: {self.profit_target}")

    def check_exit_conditions(self, current_option_price):
        """Check if we should exit the trade"""
        if not self.position_open:
            return False

        # Check stop loss
        if (
            self.position_type == "calls" and current_option_price <= self.stop_loss
        ) or (self.position_type == "puts" and current_option_price >= self.stop_loss):
            print(f"Stop loss triggered. Exiting at {current_option_price}")
            self.close_position(current_option_price)
            return True

        # Check profit target
        if (
            self.position_type == "calls" and current_option_price >= self.profit_target
        ) or (
            self.position_type == "puts" and current_option_price <= self.profit_target
        ):
            print(f"Profit target reached. Exiting at {current_option_price}")
            self.close_position(current_option_price)
            return True

        return False

    def close_position(self, exit_price):
        """Close the current position"""
        profit_loss = (
            exit_price - self.entry_price
            if self.position_type == "calls"
            else self.entry_price - exit_price
        )
        profit_pct = (profit_loss / self.entry_price) * 100

        print(
            f"Closing position at {exit_price}. P/L: ${profit_loss:.2f} ({profit_pct:.2f}%)"
        )

        self.position_open = False
        self.entry_price = 0
        self.entry_time = None
        self.position_type = None

    def run_strategy(self):
        """Main strategy loop"""
        print("Starting SPX intraminute spike detection strategy...")

        while True:
            try:
                # If we have an open position, check if we need to exit
                if self.position_open:
                    # Get current option price (in real implementation, you'd get this from API)
                    current_price = self.get_current_price()
                    chain = self.get_option_chain(
                        self.find_nearest_strike(current_price)
                    )

                    # In a real implementation, you'd look up the specific option you own
                    # This is simplified for demonstration
                    current_option_price = (
                        self.entry_price * 1.01
                    )  # Simulated price movement

                    self.check_exit_conditions(current_option_price)
                else:
                    # Look for new trade opportunities
                    df = self.get_minute_data()
                    spike_detected, direction = self.detect_spike(df)

                    if spike_detected:
                        print(f"Spike detected in {direction} direction!")
                        current_price = self.get_current_price()
                        nearest_strike = self.find_nearest_strike(current_price)

                        # Get option chain
                        chain = self.get_option_chain(nearest_strike)
                        option, option_type = self.select_option(
                            chain, direction, nearest_strike
                        )

                        # Execute the trade
                        self.execute_trade(option, option_type)

                # Wait for 1 minute before checking again
                time.sleep(60)

            except Exception as e:
                print(f"Error occurred: {e}")
                time.sleep(60)  # Wait a minute before retrying


if __name__ == "__main__":
    trader = SPXOptionIntraminuteTrader()
    trader.run_strategy()
