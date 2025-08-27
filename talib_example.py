import pandas as pd
import talib
import mplfinance as mpf
from data_downloader import DataDownloader
from dukascopy_python.instruments import INSTRUMENT_US_PLTR_US_USD, INSTRUMENT_US_AAPL_US_USD
import dukascopy_python
import pprint
from datetime import datetime

# -------------------
# Config
# -------------------
instrument = INSTRUMENT_US_AAPL_US_USD
commission = 0.0002  # 0,02%
initial_cash = 1000
trade_pct = 1  # percentuale del portafoglio da investire

df = DataDownloader.download_data_to_dataframe(
    instrument,
    interval=dukascopy_python.INTERVAL_DAY_1,
    start=datetime(2023, 1, 1),
    end=datetime.now()
)


print("Dataframe recuperato da csv:")
print(df.head())
# -------------------
# Indicatori
# -------------------
# MACD
macd, macdsignal, macdhist = talib.MACD(df["Close"], fastperiod=10, slowperiod=26, signalperiod=9)
df["MACD_Buy"] = (macd > macdsignal) & (macd.shift(1) <= macdsignal.shift(1))
df["MACD_Sell"] = (macd < macdsignal) & (macd.shift(1) >= macdsignal.shift(1))

# RSI
rsi = talib.RSI(df["Close"], timeperiod=14)
df["RSI_Buy"] = rsi < 30
df["RSI_Sell"] = rsi > 70

# Bollinger Bands
upper, middle, lower = talib.BBANDS(df["Close"], timeperiod=10, nbdevup=2, nbdevdn=2, matype=0)
df["BB_Buy"] = df["Close"] < lower
df["BB_Sell"] = df["Close"] > upper

# Engulfing
df["Engulfing_Buy"] = (df["Close"] > df["Open"]) & (df["Close"].shift(1) < df["Open"].shift(1))
df["Engulfing_Sell"] = (df["Close"] < df["Open"]) & (df["Close"].shift(1) > df["Open"].shift(1))

# OBV (On-Balance Volume)
df["OBV"] = talib.OBV(df["Close"], df["Volume"])
obv_signal = df["OBV"].diff() > 0
df["OBV_Buy"] = obv_signal & (obv_signal.shift(1) == False)
df["OBV_Sell"] = ~obv_signal & (obv_signal.shift(1) == True)

# VWAP
df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
df["VWAP_Buy"] = df["Close"] > df["VWAP"]
df["VWAP_Sell"] = df["Close"] < df["VWAP"]

# -------------------
# Funzione simulazione portafoglio
# -------------------
def simulate_portfolio(df, buy_col, sell_col, portfolio_col, stop_loss_pct=0.02, take_profit_pct=0.04):
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

# -------------------
# Simulazioni
# -------------------
report_macd = simulate_portfolio(df, "MACD_Buy", "MACD_Sell", "Portfolio_MACD")
report_rsi  = simulate_portfolio(df, "RSI_Buy", "RSI_Sell", "Portfolio_RSI")
report_bb   = simulate_portfolio(df, "BB_Buy", "BB_Sell", "Portfolio_BB")
report_eng  = simulate_portfolio(df, "Engulfing_Buy", "Engulfing_Sell", "Portfolio_Eng")
report_obv  = simulate_portfolio(df, "OBV_Buy", "OBV_Sell", "Portfolio_OBV")
report_vwap = simulate_portfolio(df, "VWAP_Buy", "VWAP_Sell", "Portfolio_VWAP")

print("\n--- REPORT PERFORMANCE ---")
print("\nMACD:"); pprint.pprint(report_macd)
print("\nRSI:"); pprint.pprint(report_rsi)
print("\nBollinger Bands:"); pprint.pprint(report_bb)
print("\nEngulfing:"); pprint.pprint(report_eng)
print("\nOBV:"); pprint.pprint(report_obv)
print("\nVWAP:"); pprint.pprint(report_vwap)

# -------------------
# Grafico con segnali BUY/SELL
# -------------------

#buy_marker = df["Close"].where(df["MACD_Buy"] | df["RSI_Buy"] | df["BB_Buy"] | df["Engulfing_Buy"] | df["OBV_Buy"] | df["VWAP_Buy"])
#sell_marker = df["Close"].where(df["MACD_Sell"] | df["RSI_Sell"] | df["BB_Sell"] | df["Engulfing_Sell"] | df["OBV_Sell"] | df["VWAP_Sell"])

buy_marker = df["Close"].where( df["BB_Buy"] )
sell_marker = df["Close"].where( df["BB_Sell"] )

apds = [
    mpf.make_addplot(macd, panel=1, color="blue", ylabel=instrument),
    mpf.make_addplot(macdsignal, panel=1, color="orange"),
    mpf.make_addplot(macdhist, panel=1, type="bar", color="gray"),
    mpf.make_addplot(df["VWAP"], color="purple", linestyle="--"),
    mpf.make_addplot(buy_marker, type="scatter", markersize=100, marker="^", color="green"),
    mpf.make_addplot(sell_marker, type="scatter", markersize=100, marker="v", color="red")
]

mpf.plot(
    df,
    type="candle",
    volume=True,
    addplot=apds,
    style="yahoo",
    title=instrument,
    show_nontrading=False
)
