from data_preparation import DataPreprocessor
from data_downloader import DataDownloader
from dukascopy_python.instruments import INSTRUMENT_US_AAPL_US_USD, INSTRUMENT_US_PLTR_US_USD
import dukascopy_python 
from datetime import datetime

instrument = INSTRUMENT_US_AAPL_US_USD

def main():
    df_history = DataDownloader.download_data_to_dataframe(
        instrument,
        interval=dukascopy_python.INTERVAL_HOUR_1,
        start=datetime(2024, 1, 1),
        end=datetime.now()
    )
    prepocessor = DataPreprocessor(df_history)
    prepocessor.generate_dataset()
    #prepocessor.generate_targets(front_up_candles=5, th_pct=0.01)

if __name__ == "__main__":
    main()