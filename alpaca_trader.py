import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
import asyncio
from datetime import datetime, timedelta, timezone
import talib
from data_downloader import DataDownloader
import mplfinance as mpf
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_US_PLTR_US_USD, INSTRUMENT_US_AAPL_US_USD

# opzioni: "1Min", "5Min", "15Min", "1Hour", "1Day"

class AlpacaTrader:
    def __init__(self, api_key, api_secret, base_url="https://paper-api.alpaca.markets", timeframe="1Day", number_of_days=365, symbol="AAPL", usa_sync=True):
        print("Initializing AlpacaTrader...")
        self.timeframe = timeframe
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.symbol = symbol
        self.number_of_days = number_of_days
        self.api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        self.account = self.api.get_account()
        self.positions = self.api.list_positions()
        self.orders = self.api.list_orders()
        self.usa_sync = usa_sync  # Sincronizza con l'orario di mercato USA (9:30 - 16:00 EST)

        print("Account status:", self.account.status)
        print("Positions:", self.positions)
        print("Portfolio value:", self.account.portfolio_value)
        print("Cash available:", self.account.cash)

    async def on_bar(self, bar):
        print("="*50)
        print(f"Nuova barra ricevuta su evento: {bar.symbol} {bar.t} Close: {bar.c}")
        print("="*50)


    async def run_trading(self):
        market_scanner = MarketPatternScanner.BollingerBands(timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        while True:
            now = datetime.now(timezone.utc)

            start = (now - timedelta(days=self.number_of_days)).date()

            start_d = start
            end_d = now.date()

            if self.usa_sync:
                start_d = datetime(start.year, start.month, start.day, 14, 30) 
                end_d = datetime(now.year, now.month, now.day, 21, 0) 


            start_date = start_d.strftime("%Y-%m-%dT%H:%M:%SZ")
            end_date = end_d.strftime("%Y-%m-%dT%H:%M:%SZ")

            print(f"Data di inzio: {start_date}")
            print(f"Data di fine: {end_date}")
            print(f"Intervallo: {self.timeframe}")

            symbol = self.symbol
            timeframe = self.timeframe
            #candles = self.get_candles(symbol, timeframe, start_date, end_date)
            #candles = DataDownloader.download_data_to_dataframe("PLTR", interval=dukascopy_python.INTERVAL_DAY_1)
            dframe = DataDownloader.download_data_to_dataframe( INSTRUMENT_US_AAPL_US_USD, interval=dukascopy_python.INTERVAL_DAY_1)

            print(dframe)

            market_scanner.apply(dframe)

            #self.elaborate_bars(bars)   
            await asyncio.sleep(60)  # ogni 60 secondi

    def get_candles(self, symbol, timeframe, start_date, end_date):

        bars = self.api.get_bars(symbol, timeframe, start=start_date, end=end_date, feed="iex").df

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

        # 2. Imposta come indice

        # Ora la colonna 'Gmt time' esiste e puoi formattarla
        #bars["Gmt time"] = pd.to_datetime(bars["Gmt time"]).dt.strftime("%d.%m.%Y %H:%M:%S.000")

        bars.set_index("Gmt time", inplace=True)

        bars.sort_index(inplace=True)
        # Mantieni solo le colonne che ti interessano
        #bars = bars[["Gmt time", "Open", "High", "Low", "Close", "Volume"]]

        # Esporta senza indice
        bars.to_csv("data/candles.csv", index=False)

        print(bars.head())

        return bars

    def elaborate_bars(self, bars):
        print("Elaborazione barre...")


    async def start_all(self):
        print("Inizio tutte le attività...")
        # Task periodico
        t1 = asyncio.create_task(self.run_trading())

        print("t1 (timed_request) avviata")


        await t1



def plot_results(df, buy_clm, sell_clm):
    buy_marker = df["Close"].where( buy_clm )
    sell_marker = df["Close"].where( sell_clm )

    apds = [
        mpf.make_addplot(buy_marker, type="scatter", markersize=100, marker="^", color="green"),
        mpf.make_addplot(sell_marker, type="scatter", markersize=100, marker="v", color="red")
    ]

    mpf.plot(
        df,
        type="candle",
        volume=True,
        addplot=apds,
        style="yahoo",
        title="Chart",
        show_nontrading=False
    )

class MarketPatternScanner:
    
    class BollingerBands:
        def __init__(self, timeperiod=10, nbdevup=2, nbdevdn=2, matype=0):
            self.timeperiod = timeperiod
            self.nbdevup = nbdevup
            self.nbdevdn = nbdevdn
            self.matype = matype    

        def apply(self, df):
            upper, middle, lower = talib.BBANDS(df["Close"], timeperiod=self.timeperiod, nbdevup=self.nbdevup, nbdevdn=self.nbdevdn , matype = self.matype)
            df["BB_Buy"] = df["Close"] < lower
            df["BB_Sell"] = df["Close"] > upper
            print("Bollinger Bands signals:")
            pd.set_option('display.max_rows', None)
            print(df)
            plot_results(df, df["BB_Buy"], df["BB_Sell"])

        def run_optimization(self, df):
            print("Running optimization...")
            # Qui puoi implementare la logica di ottimizzazione
            # Ad esempio, testare diverse combinazioni di parametri per i segnali
            # e valutare le performance del portafoglio

    class Statistics:
        def simulate_portfolio(df, buy_col, sell_col, portfolio_col, stop_loss_pct=0.02, take_profit_pct=0.04, initial_cash=1000, commission=0.0002, trade_pct=0.2):
            """
            Simula un portafoglio basato sui segnali di buy/sell.
            
            stop_loss_pct e take_profit_pct devono essere numeri decimali (es. 0.02 = 2%)
            """
            cash = initial_cash
            position = 0.0
            entry_price = 0.0
            portfolio = []
            trades = []

            for i in range(len(df)):
                price = df["Close"].iloc[i]

                # Controllo stop loss / take profit
                if position > 0:
                    if stop_loss_pct is not None and price <= entry_price * (1 - stop_loss_pct):
                        # Stop loss
                        exit_value = position * price * (1 - commission)
                        trades.append(exit_value - entry_price * position)
                        cash += exit_value
                        position = 0.0
                        entry_price = 0.0

                    elif take_profit_pct is not None and price >= entry_price * (1 + take_profit_pct):
                        # Take profit
                        exit_value = position * price * (1 - commission)
                        trades.append(exit_value - entry_price * position)
                        cash += exit_value
                        position = 0.0
                        entry_price = 0.0

                # Acquisto
                if df[buy_col].iloc[i] and position == 0:
                    invest_amount = cash * trade_pct
                    if invest_amount > 0:
                        position = invest_amount / price * (1 - commission)
                        entry_price = price
                        cash -= invest_amount

                # Vendita
                elif df[sell_col].iloc[i] and position > 0:
                    exit_value = position * price * (1 - commission)
                    trades.append(exit_value - entry_price * position)
                    cash += exit_value
                    position = 0.0
                    entry_price = 0.0

                # Aggiorno il valore del portafoglio ad ogni riga
                portfolio.append(cash + position * price)

            # Aggiorno il DataFrame
            df[portfolio_col] = portfolio