from datetime import datetime, timedelta
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_GBP_USD, INSTRUMENT_US_PLTR_US_USD
import pandas as pd
import yfinance as yf

class DataDownloader:
    def download_data_to_csv(instrument, start = datetime(2025, 1, 1), end= datetime.now(), interval=dukascopy_python.INTERVAL_DAY_1):
        print(f"Downloading data for {instrument} from {start} to {end} with interval {interval}")
        selected_instrument = instrument

        offer_side = dukascopy_python.OFFER_SIDE_BID

        df = dukascopy_python.fetch(
            selected_instrument,
            interval,
            offer_side,
            start,
            end,
        )

        df = df.reset_index()

        # Rinomina le colonne
        df.rename(columns={
            "timestamp": "Gmt time",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }, inplace=True)
        

        # Converti la colonna 'Gmt time' nel formato richiesto
        df["Gmt time"] = pd.to_datetime(df["Gmt time"]).dt.strftime("%d.%m.%Y %H:%M:%S.000")

        # Salva in CSV
        df.to_csv(f"data/{selected_instrument.split('.')[0]}.csv", index=False, date_format="%d.%m.%Y %H:%M:%S")

    def download_data_to_csv_yf(symbol, start=datetime(2020, 1, 1), end=datetime.now(), interval='1d'):
        """
        Scarica dati storici da Yahoo Finance e li salva in CSV con formato simile a Dukascopy.
        
        symbol: str -> simbolo dello strumento (es. "EURUSD=X")
        start: datetime -> data di inizio
        end: datetime -> data di fine
        interval: str -> intervallo ('1d', '1h', '5m', ecc.)
        """
        print(f"Downloading data for {symbol} from {start} to {end} with interval {interval}")
        
        df = yf.download(symbol, start=start, end=end, interval=interval)
        print("Data downloaded successfully.")

        df = df.reset_index()
        
        df.rename(columns={
            "Date": "Gmt time",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        }, inplace=True)

        df["Gmt time"] = pd.to_datetime(df["Gmt time"]).dt.strftime("%d.%m.%Y %H:%M:%S.000")

        df.to_csv(f"data/{symbol.replace('=','').split('.')[0]}.csv", index=False)
        print(f"CSV saved as data/{symbol.replace('=','').split('.')[0]}.csv")