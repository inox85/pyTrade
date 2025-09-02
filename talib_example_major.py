import pandas as pd
import talib
import mplfinance as mpf
from data_downloader import DataDownloader
from dukascopy_python.instruments import INSTRUMENT_US_AAPL_US_USD
import dukascopy_python
import pprint
from datetime import datetime
# ===========================
#  Download dati
# ===========================

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

# ===========================
#  Indicatori tecnici
# ===========================
# Bollinger Bands
upper, middle, lower = talib.BBANDS(df["Close"], timeperiod=20)

# RSI
rsi = talib.RSI(df["Close"], timeperiod=14)

# MACD
macd, macdsignal, macdhist = talib.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)

# OBV
obv = talib.OBV(df["Close"], df["Volume"])

# VWAP (semplificato)
df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3
df["VWAP"] = (df["TP"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

# MFI
mfi = talib.MFI(df["High"], df["Low"], df["Close"], df["Volume"], timeperiod=14)

# Volume medio
sma_volume = df["Volume"].rolling(20).mean()

# ===========================
#  Strategie di trading
# ===========================
# Bollinger Bands semplice
df["BB_Buy"] = df["Close"] <= lower
df["BB_Sell"] = df["Close"] >= upper

# RSI
df["RSI_Buy"] = rsi < 30
df["RSI_Sell"] = rsi > 70

# MACD
df["MACD_Buy"] = macd > macdsignal
df["MACD_Sell"] = macd < macdsignal

# Engulfing (pattern candele)
engulfing = talib.CDLENGULFING(df["Open"], df["High"], df["Low"], df["Close"])
df["Engulfing_Buy"] = engulfing > 0
df["Engulfing_Sell"] = engulfing < 0

# OBV (trend up/down)
df["OBV_Buy"] = obv > obv.shift(1)
df["OBV_Sell"] = obv < obv.shift(1)

# VWAP
df["VWAP_Buy"] = df["Close"] > df["VWAP"]
df["VWAP_Sell"] = df["Close"] < df["VWAP"]

# MFI
df["MFI_Buy"] = mfi < 20
df["MFI_Sell"] = mfi > 80

# Volume anomalo
df["VolAnomaly_Buy"] = df["Volume"] > (sma_volume * 1.5)
df["VolAnomaly_Sell"] = df["Volume"] < (sma_volume * 0.5)

# --- NUOVA STRATEGIA: Bollinger + RSI + Volume ---
df["BB_RSI_Buy"] = (df["Close"] <= lower) & (rsi < 30) #& (df["Volume"] > sma_volume)
df["BB_RSI_Sell"] = (df["Close"] >= upper) & (rsi > 70) #& (df["Volume"] > sma_volume)

# ===========================
#  Funzione di simulazione
# ===========================
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

# ===========================
#  Backtest di tutte le strategie
# ===========================
report_bb = simulate_portfolio(df, "BB_Buy", "BB_Sell", "Portfolio_BB")
report_rsi = simulate_portfolio(df, "RSI_Buy", "RSI_Sell", "Portfolio_RSI")
report_macd = simulate_portfolio(df, "MACD_Buy", "MACD_Sell", "Portfolio_MACD")
report_engulfing = simulate_portfolio(df, "Engulfing_Buy", "Engulfing_Sell", "Portfolio_Engulfing")
report_obv = simulate_portfolio(df, "OBV_Buy", "OBV_Sell", "Portfolio_OBV")
report_vwap = simulate_portfolio(df, "VWAP_Buy", "VWAP_Sell", "Portfolio_VWAP")
report_mfi = simulate_portfolio(df, "MFI_Buy", "MFI_Sell", "Portfolio_MFI")
report_vol = simulate_portfolio(df, "VolAnomaly_Buy", "VolAnomaly_Sell", "Portfolio_Vol")
report_bb_rsi = simulate_portfolio(df, "BB_RSI_Buy", "BB_RSI_Sell", "Portfolio_BB_RSI")

# ===========================
#  Risultati
# ===========================
print("\n📊 RISULTATI STRATEGIE")
print("Bollinger Bands:"); pprint.pprint(report_bb)
print("\nRSI:"); pprint.pprint(report_rsi)
print("\nMACD:"); pprint.pprint(report_macd)
print("\nEngulfing:"); pprint.pprint(report_engulfing)
print("\nOBV:"); pprint.pprint(report_obv)
print("\nVWAP:"); pprint.pprint(report_vwap)
print("\nMFI:"); pprint.pprint(report_mfi)
print("\nVolume Anomaly:"); pprint.pprint(report_vol)
print("\n✅ Bollinger + RSI + Volume:"); pprint.pprint(report_bb_rsi)

# ===========================
#  Grafico con segnali
# ===========================

apds = [
    mpf.make_addplot(upper, color="red"),
    mpf.make_addplot(middle, color="blue"),
    mpf.make_addplot(lower, color="green"),
    mpf.make_addplot(rsi, panel=1, color="purple", ylabel="RSI"),
    mpf.make_addplot(macd, panel=2, color="blue", ylabel="MACD"),
    mpf.make_addplot(macdsignal, panel=2, color="red"),
    mpf.make_addplot(obv, panel=3, color="orange", ylabel="OBV"),
    mpf.make_addplot(df["VWAP"], color="black"),
    mpf.make_addplot(mfi, panel=4, color="brown", ylabel="MFI"),
]

# Marker per segnali Bollinger+RSI

bb_rsi_buy_marker = df["Close"].where(df["BB_RSI_Buy"])
bb_rsi_sell_marker = df["Close"].where(df["BB_RSI_Sell"])

if bb_rsi_buy_marker.notna().any():
    apds.append(mpf.make_addplot(bb_rsi_buy_marker, type="scatter", markersize=80, marker="^", color="lime"))

if bb_rsi_sell_marker.notna().any():
    apds.append(mpf.make_addplot(bb_rsi_sell_marker, type="scatter", markersize=80, marker="v", color="darkred"))

mpf.plot(df, type="candle", style="yahoo", addplot=apds, volume=True, title=f"{instrument} - Strategie Trading")
