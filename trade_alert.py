import pandas as pd
import talib
import time
from data_downloader import DataDownloader
from dukascopy_python.instruments import INSTRUMENT_US_PLTR_US_USD
import dukascopy_python
# -------------------
# Config
# -------------------
check_interval_seconds = 3600  # controlla ogni ora
commission = 0.0002
initial_cash = 10000

# -------------------
# Funzione per aggiornare dati
# -------------------
def update_data():
    # Scarica/aggiorna il CSV più recente
    DataDownloader.download_data_to_csv(INSTRUMENT_US_PLTR_US_USD, interval=dukascopy_python.INTERVAL_HOUR_1)

    df = pd.read_csv(
        "data/PLTR.csv",
        index_col="Gmt time",
        parse_dates=["Gmt time"],
        dayfirst=True
    )

    df = df[~df.index.isna()]
    return df

# -------------------
# Funzione per calcolare segnali
# -------------------
def compute_signals(df):
    # MACD
    macd, macdsignal, macdhist = talib.MACD(df["Close"], fastperiod=12, slowperiod=26, signalperiod=9)
    df["MACD_Buy"] = (macd > macdsignal) & (macd.shift(1) <= macdsignal.shift(1))
    df["MACD_Sell"] = (macd < macdsignal) & (macd.shift(1) >= macdsignal.shift(1))

    # RSI
    rsi = talib.RSI(df["Close"], timeperiod=14)
    df["RSI_Buy"] = rsi < 30
    df["RSI_Sell"] = rsi > 70

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(df["Close"], timeperiod=20)
    df["BB_Buy"] = df["Close"] < lower
    df["BB_Sell"] = df["Close"] > upper

    return df

# -------------------
# Funzione per controllare segnali e inviare alert
# -------------------
def check_alerts(df):
    latest = df.iloc[-1]

    alerts = []

    if latest["MACD_Buy"]:
        alerts.append("MACD: BUY")
    if latest["MACD_Sell"]:
        alerts.append("MACD: SELL")
    
    if latest["RSI_Buy"]:
        alerts.append("RSI: BUY")
    if latest["RSI_Sell"]:
        alerts.append("RSI: SELL")
    
    if latest["BB_Buy"]:
        alerts.append("Bollinger: BUY")
    if latest["BB_Sell"]:
        alerts.append("Bollinger: SELL")

    if alerts:
        print(f"\nAlert per {latest.name}:")
        for a in alerts:
            print(f"  - {a}")
    else:
        print(f"\nNessun segnale per {latest.name}")

# -------------------
# Loop principale
# -------------------
while True:
    try:
        df = update_data()
        df = compute_signals(df)
        check_alerts(df)
    except Exception as e:
        print("Errore durante l'update o il calcolo:", e)

    print(f"\nProssimo controllo tra {check_interval_seconds/60:.0f} minuti...\n")
    time.sleep(check_interval_seconds)
