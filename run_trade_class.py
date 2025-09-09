from data_preprocessor import DataPreprocessor
from data_downloader import DataDownloader
from dukascopy_python.instruments import INSTRUMENT_US_AAPL_US_USD, INSTRUMENT_US_PLTR_US_USD
import dukascopy_python 
from datetime import datetime, timedelta, timezone
from alpaca_service import AlpacaService


instruments = [INSTRUMENT_US_AAPL_US_USD, INSTRUMENT_US_PLTR_US_USD]

symbols = ["TSLA", "MSFT", "PLTR", "AAPL", "NVDA", "SPY"]

def main():
    for symbol in symbols:
        alpaca = AlpacaService(symbol=symbol)

        now = datetime.now(timezone.utc)

        start = (now - timedelta(days=365)).date()

        start_d = datetime(start.year, start.month, start.day, 14, 30) 
        end_d = datetime(now.year, now.month, now.day, 21, 0) 

        start_date = start_d.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_date = end_d.strftime("%Y-%m-%dT%H:%M:%SZ")

        print("Dowload data per:", symbol)

        df_history = alpaca.download_data("1H", start_date=start_date, end_date=end_date )

        preprocessor = DataPreprocessor(df_history, symbol=symbol, interval="1H")

        df = preprocessor.get_dataset()

        print("Database di origine:", df)

        df_tecnical = preprocessor.generate_tecnical_dataset(df)

        print("Database di tecnico:", df_tecnical)

        df_targets = preprocessor.generate_targets_dataset(df_tecnical)
        print("Database di targets:", df)

        df.to_csv(f"data/{symbol}_processed.csv", sep=';', encoding='utf-8')

        print(df_targets)



if __name__ == "__main__":      
    main()
