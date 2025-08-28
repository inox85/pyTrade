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
    
    def download_data_to_csv_yf(instrument, start=datetime(2025, 1, 1), end=datetime.now(), interval="1d"):
        print(f"Downloading data for {instrument} from {start} to {end} with interval {interval}")

        all_data = []
        chunk_start = start

        # Scarica a blocchi di massimo 1 mese
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=30), end)
            print(f"  - Downloading chunk {chunk_start} → {chunk_end}")

            df = yf.download(
                instrument,
                start=chunk_start,
                end=chunk_end,
                interval=interval,
                progress=False
            )

            if not df.empty:
                df.reset_index(inplace=True)
                all_data.append(df)

            chunk_start = chunk_end

        if not all_data:
            print("No data downloaded.")
            return None

        # Unisci i dati scaricati
        df = pd.concat(all_data, ignore_index=True)

        # Rinomina le colonne
        df.rename(columns={
            "Date": "Gmt time",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume"
        }, inplace=True)

        # Converte la colonna 'Gmt time' nel formato richiesto
        df["Gmt time"] = pd.to_datetime(df["Gmt time"]).dt.strftime("%d.%m.%Y %H:%M:%S.000")

        file_path = f"data/{instrument.replace('=','_')}.csv"

        # Salva in CSV
        df.to_csv(file_path, index=False, date_format="%d.%m.%Y %H:%M:%S")

        return file_path
    
    def download_data_to_dataframe_yf(instrument, start=datetime(2025, 1, 1), end=datetime.now(), interval="1d"):
        """
        Scarica dati finanziari in blocchi mensili usando yfinance.

        Args:
            instrument (str): Il simbolo dello strumento finanziario (es. "AAPL" per Apple).
            start (datetime): La data di inizio.
            end (datetime): La data di fine.
            interval (str): L'intervallo dei dati (es. "1d", "1h", "1m").

        Returns:
            pd.DataFrame: Un DataFrame con i dati storici, o un DataFrame vuoto se non ne trova.
        """
        instrument = "PLTR"
        print(f"Downloading data for {instrument} from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} with interval {interval}")

        all_dfs = []
        current_start = start

        # Cicla a blocchi di 30 giorni
        while current_start < end:
            current_end = min(current_start + timedelta(days=30), end)
            
            print(f"  > Downloading chunk from {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
            
            df_chunk = yf.download(instrument, start=current_start, end=current_end, interval=interval, progress=False)

            if not df_chunk.empty:
                all_dfs.append(df_chunk)

            current_start = current_end + timedelta(days=1)

        if all_dfs:
            df = pd.concat(all_dfs)
            
            # Rinomina le colonne per coerenza
            df.rename(columns={
                "Adj Close": "Adj Close",
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume"
            }, inplace=True)
            
            # L'indice è già un DatetimeIndex, rinominiamolo
            df.index.name = "Gmt time"
            
            # Rimuove duplicati se ci sono sovrapposizioni tra i chunk
            df = df[~df.index.duplicated(keep='first')]

            return df

        print("No data found for the specified instrument and date range.")
        return pd.DataFrame()