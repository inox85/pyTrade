from datetime import datetime, timedelta
import dukascopy_python
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_GBP_USD, INSTRUMENT_US_PLTR_US_USD
import pandas as pd
#import yfinance as yf

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
        file_path = f"data/{selected_instrument.split('.')[0]}.csv"
        # Salva in CSV
        df.to_csv(file_path, index=False, date_format="%d.%m.%Y %H:%M:%S")

        return file_path
    
    def download_data_to_dataframe(instrument, start = datetime(2025, 1, 1), end= datetime.now(), interval=dukascopy_python.INTERVAL_DAY_1):
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
        
        df["Gmt time"] = pd.to_datetime(df["Gmt time"]).dt.strftime("%d.%m.%Y %H:%M:%S.000")

        df.to_csv("data/temp.csv", index=False, date_format="%d.%m.%Y %H:%M:%S")

        df = pd.read_csv(
            "data/temp.csv",
            index_col="Gmt time",
            parse_dates=["Gmt time"],
            dayfirst=True
        )

        df = df[~df.index.isna()]

        return df
