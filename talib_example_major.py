import pandas as pd
import talib
import mplfinance as mpf
from data_downloader import DataDownloader
from dukascopy_python.instruments import INSTRUMENT_US_AAPL_US_USD
import dukascopy_python
import pprint
from datetime import datetime

# --- Parametri generali ---
instrument = INSTRUMENT_US_AAPL_US_USD
commission = 0.0002  # 0,02%
initial_cash = 1000
trade_pct = 1  # percentuale del portafoglio da investire

# --- Download dati ---
df = DataDownloader.download_data_to_dataframe(
    instrument,
    interval=dukascopy_python.INTERVAL_HOUR_1,
    start=datetime(2025, 7, 28),
    end=datetime.now()
)

print("Dataframe recuperato da csv:")
print(df.head())

# -------------------
# CALCOLO INDICATORI TECNICI
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
upper, middle, lower = talib.BBANDS(df["Close"], timeperiod=10, nbdevup=2.77, nbdevdn=2.5, matype=4)
df["BB_Buy"] = df["Close"] < lower
df["BB_Sell"] = df["Close"] > upper

# Engulfing
df["Engulfing_Buy"] = (df["Close"] > df["Open"]) & (df["Close"].shift(1) < df["Open"].shift(1))
df["Engulfing_Sell"] = (df["Close"] < df["Open"]) & (df["Close"].shift(1) > df["Open"].shift(1))

# OBV
df["OBV"] = talib.OBV(df["Close"], df["Volume"])
obv_signal = df["OBV"].diff() > 0
df["OBV_Buy"] = obv_signal & (obv_signal.shift(1) == False)
df["OBV_Sell"] = ~obv_signal & (obv_signal.shift(1) == True)

# VWAP
df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
df["VWAP_Buy"] = df["Close"] > df["VWAP"]
df["VWAP_Sell"] = df["Close"] < df["VWAP"]

# Money Flow Index (MFI)
mfi = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
df['MFI_Buy'] = mfi < 20
df['MFI_Sell'] = mfi > 80

# Volume Anomaly
sma_volume = talib.SMA(df['Volume'], timeperiod=20)
volume_spike = df['Volume'] > (sma_volume * 2.0)
df['VolAnomaly_Buy'] = volume_spike & (df['Close'] > df['Open'])
df['VolAnomaly_Sell'] = volume_spike & (df['Close'] < df['Open'])

# -------------------
# Breakout con conferma volumi + ADX
# -------------------
df["Donchian_High"] = df["High"].rolling(window=20).max()
df["Donchian_Low"] = df["Low"].rolling(window=20).min()
df["Vol_MA20"] = talib.SMA(df["Volume"], timeperiod=20)
df["ADX"] = talib.ADX(df["High"], df["Low"], df["Close"], timeperiod=14)
adx_threshold = 20

df["Breakout_Up"] = (
    (df["Close"] > df["Donchian_High"].shift(1)) &
    (df["Volume"] > df["Vol_MA20"]) &
    (df["ADX"] > adx_threshold)
)

df["Breakout_Down"] = (
    (df["Close"] < df["Donchian_Low"].shift(1)) &
    (df["Volume"] > df["Vol_MA20"]) &
    (df["ADX"] > adx_threshold)
)

# -------------------
# Funzione simulazione portafoglio
# -------------------
def simulate_portfolio(df, buy_col, sell_col, portfolio_col, stop_loss_pct=0.02, take_profit_pct=0.04):
    cash = initial_cash
    position = 0.0
    entry_price = 0.0
    portfolio = []
    trades = []

    for i in range(len(df)):
        price = df["Close"].iloc[i]

        # Stop loss / take profit
        if position > 0:
            if stop_loss_pct and price <= entry_price * (1 - stop_loss_pct):
                exit_value = position * price * (1 - commission)
                trades.append(exit_value - (position * entry_price))
                cash += exit_value
                position = 0
                entry_price = 0
            elif take_profit_pct and price >= entry_price * (1 + take_profit_pct):
                exit_value = position * price * (1 - commission)
                trades.append(exit_value - (position * entry_price))
                cash += exit_value
                position = 0
                entry_price = 0

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
            trades.append(exit_value - (position * entry_price))
            cash += exit_value
            position = 0
            entry_price = 0

        portfolio.append(cash + position * price)

    df[portfolio_col] = portfolio
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
reports = {
    "MACD": simulate_portfolio(df, "MACD_Buy", "MACD_Sell", "Portfolio_MACD"),
    "RSI": simulate_portfolio(df, "RSI_Buy", "RSI_Sell", "Portfolio_RSI"),
    "Bollinger Bands": simulate_portfolio(df, "BB_Buy", "BB_Sell", "Portfolio_BB"),
    "Engulfing": simulate_portfolio(df, "Engulfing_Buy", "Engulfing_Sell", "Portfolio_Eng"),
    "OBV": simulate_portfolio(df, "OBV_Buy", "OBV_Sell", "Portfolio_OBV"),
    "VWAP": simulate_portfolio(df, "VWAP_Buy", "VWAP_Sell", "Portfolio_VWAP"),
    "MFI": simulate_portfolio(df, "MFI_Buy", "MFI_Sell", "Portfolio_MFI"),
    "Volume Anomaly": simulate_portfolio(df, "VolAnomaly_Buy", "VolAnomaly_Sell", "Portfolio_VolAnomaly"),
    "Breakout": simulate_portfolio(df, "Breakout_Up", "Breakout_Down", "Portfolio_Breakout")
}

# -------------------
# Stampa report
# -------------------
print("\n--- REPORT PERFORMANCE ---")
for k, v in reports.items():
    print(f"\n{k}:")
    pprint.pprint(v)

# -------------------
# PLOT mplfinance
# -------------------
buy_marker = df["Close"].where(df["BB_Buy"])
sell_marker = df["Close"].where(df["BB_Sell"])

buy_breakout = df["Close"].where(df["Breakout_Up"])
sell_breakout = df["Close"].where(df["Breakout_Down"])

apds = [
    mpf.make_addplot(macd, panel=2, color="blue", ylabel="MACD"),
    mpf.make_addplot(macdsignal, panel=2, color="orange"),
    mpf.make_addplot(macdhist, panel=2, type="bar", color="gray"),
    mpf.make_addplot(mfi, panel=3, color='purple', ylabel='MFI'),
    mpf.make_addplot([80]*len(df), panel=3, color='r', linestyle='--'),
    mpf.make_addplot([20]*len(df), panel=3, color='g', linestyle='--'),
    mpf.make_addplot(df["VWAP"], color="purple", linestyle="--"),
]

# Aggiungi segnali Bollinger
if buy_marker.notna().any():
    apds.append(mpf.make_addplot(buy_marker, type="scatter", markersize=100, marker="^", color="green"))
if sell_marker.notna().any():
    apds.append(mpf.make_addplot(sell_marker, type="scatter", markersize=100, marker="v", color="red"))

# Aggiungi segnali breakout
if buy_breakout.notna().any():
    apds.append(mpf.make_addplot(buy_breakout, type="scatter", markersize=100, marker="^", color="lime"))
if sell_breakout.notna().any():
    apds.append(mpf.make_addplot(sell_breakout, type="scatter", markersize=100, marker="v", color="magenta"))

mpf.plot(
    df,
    type="candle",
    volume=True,
    addplot=apds,
    style="yahoo",
    title=f"{instrument} Signals & Breakouts",
    show_nontrading=False,
    panel_ratios=(6, 1, 2, 2)
)
