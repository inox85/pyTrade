import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
import asyncio
from datetime import datetime, timedelta, timezone
import talib
from data_downloader import DataDownloader
import mplfinance as mpf
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_US_PLTR_US_USD, INSTRUMENT_US_AAPL_US_USD, INSTRUMENT_US_NVDA_US_USD, INSTRUMENT_US_MSFT_US_USD, INSTRUMENT_US_QCOM_US_USD, INSTRUMENT_US_INTC_US_USD
import random

class AlpacaService:    
    def __init__(self, api_key="PK4TFTC9YYDHD5S51QNM", api_secret="10KDzFIhvTokJUnMStGCW6IioaEUJaEdCoJch7qo", symbol="AAPL", usa_sync=True):
        self.base_url = "https://paper-api.alpaca.markets"
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.usa_sync = usa_sync
        self.api = tradeapi.REST(api_key, api_secret, self.base_url, api_version='v2')
        self.account = self.api.get_account()
        self.positions = self.api.list_positions()
        self.orders = self.api.list_orders()

        print("Account status:", self.account.status)
        print("Positions:", self.positions)
        print("Portfolio value:", self.account.portfolio_value)
        print("Cash available:", self.account.cash)
    
    def download_data(self, timeframe, start_date, end_date):
        bars = self.api.get_bars(self.symbol, timeframe, start=start_date, end=end_date, feed="iex").df
        bars = bars.reset_index()

        # Verifica nomi colonne disponibili
        print(bars.columns)

        # Rinominare la colonna 'timestamp' in 'Gmt time'
        bars = bars.rename(columns={
            "timestamp": "Gmt time",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })

        print(bars.head())

        bars["Gmt time"] = pd.to_datetime(bars["Gmt time"])
        bars.sort_values("Gmt time", inplace=True)

        # Mantieni solo le colonne che ti interessano
        bars = bars[["Gmt time", "Open", "High", "Low", "Close", "Volume", "trade_count", "vwap"]]

        # Esporta senza indice
        bars.to_csv("data/candles.csv", index=False)

        return bars

        

