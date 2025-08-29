import pandas as pd
import talib
import mplfinance as mpf
from data_downloader import DataDownloader
from dukascopy_python.instruments import INSTRUMENT_US_AAPL_US_USD
import dukascopy_python
import pprint
from datetime import datetime

instrument = INSTRUMENT_US_AAPL_US_USD
commission = 0.0002  # 0,02%
initial_cash = 1000
trade_pct = 1  # percentuale del portafoglio da investire

df = DataDownloader.download_data_to_dataframe(
    instrument,
    interval=dukascopy_python.INTERVAL_DAY_1,
    start=datetime(2022, 8, 28),
    end=datetime.now()
)

print("Dataframe recuperato da csv:")
print(df.head())

# --- Calcolo Indicatori Tecnici ---

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

# --- NUOVI INDICATORI BASATI SUI VOLUMI ---

# 1. Money Flow Index (MFI)
mfi = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
df['MFI_Buy'] = mfi < 20  # Ipervenduto
df['MFI_Sell'] = mfi > 80 # Ipercomprato

# 2. Anomalia di Volume (Volume Spike)
sma_volume = talib.SMA(df['Volume'], timeperiod=20)
volume_spike = df['Volume'] > (sma_volume * 2.0) # Il volume è più del doppio della sua media a 20 periodi
df['VolAnomaly_Buy'] = volume_spike & (df['Close'] > df['Open'])  # Spike di volume su candela verde
df['VolAnomaly_Sell'] = volume_spike & (df['Close'] < df['Open']) # Spike di volume su candela rossa


# -------------------
# Funzione simulazione portafoglio (INVARIATA)
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
                trades.append(exit_value - (position * entry_price)) # Calcolo profitto/perdita
                cash += exit_value
                position = 0.0
                entry_price = 0.0

            elif take_profit_pct is not None and price >= entry_price * (1 + take_profit_pct):
                # Take profit
                exit_value = position * price * (1 - commission)
                trades.append(exit_value - (position * entry_price)) # Calcolo profitto/perdita
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
            trades.append(exit_value - (position * entry_price)) # Calcolo profitto/perdita
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

report_macd = simulate_portfolio(df, "MACD_Buy", "MACD_Sell", "Portfolio_MACD")
report_rsi  = simulate_portfolio(df, "RSI_Buy", "RSI_Sell", "Portfolio_RSI")
report_bb   = simulate_portfolio(df, "BB_Buy", "BB_Sell", "Portfolio_BB")
report_eng  = simulate_portfolio(df, "Engulfing_Buy", "Engulfing_Sell", "Portfolio_Eng")
report_obv  = simulate_portfolio(df, "OBV_Buy", "OBV_Sell", "Portfolio_OBV")
report_vwap = simulate_portfolio(df, "VWAP_Buy", "VWAP_Sell", "Portfolio_VWAP")
report_mfi = simulate_portfolio(df, "MFI_Buy", "MFI_Sell", "Portfolio_MFI")
report_vol_anomaly = simulate_portfolio(df, "VolAnomaly_Buy", "VolAnomaly_Sell", "Portfolio_VolAnomaly")

print("\n--- REPORT PERFORMANCE ---")
print("\nMACD:"); pprint.pprint(report_macd)
print("\nRSI:"); pprint.pprint(report_rsi)
print("\nBollinger Bands:"); pprint.pprint(report_bb)
print("\nEngulfing:"); pprint.pprint(report_eng)
print("\nOBV:"); pprint.pprint(report_obv)
print("\nVWAP:"); pprint.pprint(report_vwap)
# --- NUOVI REPORT ---
print("\nMoney Flow Index (MFI):"); pprint.pprint(report_mfi)
print("\nVolume Anomaly:"); pprint.pprint(report_vol_anomaly)


# Esempio per visualizzare i segnali di uno dei nuovi indicatori (Bollinger Bands)
buy_marker = df["Close"].where(df["BB_Buy"])
sell_marker = df["Close"].where(df["BB_Sell"])

# Lista base per i pannelli aggiuntivi
apds = [
    # Pannello 2: MACD
    mpf.make_addplot(macd, panel=2, color="blue", ylabel="MACD"),
    mpf.make_addplot(macdsignal, panel=2, color="orange"),
    mpf.make_addplot(macdhist, panel=2, type="bar", color="gray"),
    
    # Pannello 3: MFI
    mpf.make_addplot(mfi, panel=3, color='purple', ylabel='MFI'),
    mpf.make_addplot([80]*len(df), panel=3, color='r', linestyle='--'),
    mpf.make_addplot([20]*len(df), panel=3, color='g', linestyle='--'),
    
    # Linea VWAP sul grafico principale
    mpf.make_addplot(df["VWAP"], color="purple", linestyle="--"),
]

# --- CONTROLLO DI SICUREZZA ---
# Aggiungi i segnali al grafico SOLO SE esistono dei segnali validi

if buy_marker.notna().any():
    print("Trovati segnali di acquisto da plottare.")
    apds.append(mpf.make_addplot(buy_marker, type="scatter", markersize=100, marker="^", color="green"))

if sell_marker.notna().any():
    print("Trovati segnali di vendita da plottare.")
    apds.append(mpf.make_addplot(sell_marker, type="scatter", markersize=100, marker="v", color="red"))

# Ora esegui il plot
mpf.plot(
    df,
    type="candle",
    volume=True,
    addplot=apds,
    style="yahoo",
    title=f"{instrument} with MFI signals",
    show_nontrading=False,
    panel_ratios=(6, 1, 2, 2) # (principale, volume, macd, mfi)
)
