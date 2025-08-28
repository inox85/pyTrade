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

    async def optimize(self):
        print("Running optimization...")
        market_scanner = MarketPatternScanner.BollingerBands(timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

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
        dframe = DataDownloader.download_data_to_dataframe( INSTRUMENT_US_QCOM_US_USD, interval=dukascopy_python.INTERVAL_HOUR_1, start=datetime(2025,7,26), end=datetime.now())

        print(dframe)

        market_scanner.run_optimization(dframe)

        #self.elaborate_bars(bars)   

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

            dframe = DataDownloader.download_data_to_dataframe(INSTRUMENT_US_AAPL_US_USD, interval=dukascopy_python.INTERVAL_HOUR_1)

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

    async def start_optimization(self):
        print("Inizio tutte le attività...")
        # Task periodico
        t1 = asyncio.create_task(self.optimize())

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
            wrate = 0
            profit = 0
            from tqdm import tqdm
            for i in tqdm(range(100000)):
                period = random.randint(8, 15)
                nbup = random.uniform(1.5, 3.0)
                nbdn = random.uniform(1.5, 3.0)
                stop_loss_pct = random.uniform(0.01, 0.1)
                coeff = random.uniform(1.0, 3.0)
                take_profit_pct = coeff * stop_loss_pct
                for matype in range(0, 8):
                    upper, middle, lower = talib.BBANDS(df["Close"], timeperiod=period, nbdevup=nbup, nbdevdn=nbdn , matype=matype)
                    df["BB_Buy"] = df["Close"] < lower
                    df["BB_Sell"] = df["Close"] > upper
                    report = MarketPatternScanner.Statistics.simulate_portfolio(df, "BB_Buy", "BB_Sell", "Portfolio", stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct)
                    if report["Final Portfolio"] > profit:
                        profit = report["Final Portfolio"]
                        print()
                        print(f"New best profit: {profit:.2f} with params: period={period}, nbdevup={nbup:.2f}, nbdevdn={nbdn:.2f}, matype={matype}, stop_loss_pct={stop_loss_pct:.2f}, take_profit_pct={take_profit_pct:.2f}")
                        print()
            print(f"Best profit after optimization: {profit:.2f}")

    class Statistics:
        def simulate_portfolio(df, buy_col, sell_col, portfolio_col, stop_loss_pct=0.03, take_profit_pct=0.6, initial_cash=100, commission=0.0002, trade_pct=1):
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

            
    # Statistiche
            final_portfolio = cash + position * df["Close"].iloc[-1]
            roi = (final_portfolio - initial_cash) / initial_cash * 100
            num_trades = len(trades)
            wins = len([t for t in trades if t > 0])
            losses = len([t for t in trades if t <= 0])
            win_rate = wins / num_trades * 100 if num_trades > 0 else 0
            avg_profit = sum(trades) / num_trades if num_trades > 0 else 0
            max_drawdown = (df[portfolio_col].cummax() - df[portfolio_col]).max()

            return {
                "Final Portfolio": final_portfolio,
                "ROI %": roi,
                "Number of Trades": num_trades,
                "Wins": wins,
                "Losses": losses,
                "Win rate %": win_rate,
                "Average Profit per Trade": avg_profit,
                "Max Drawdown": max_drawdown
            }