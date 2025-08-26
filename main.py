#from trade_anayzer import TradeAnalyzer
from dukascopy_python.instruments import INSTRUMENT_US_PLTR_US_USD, INSTRUMENT_VCCY_ETH_USD
from data_downloader import DataDownloader
from candlestick_analyzer import CandlestickAnalyzer, Strategy_02, Strategy_01, run_backtest

#DataDownloader.download_data_to_csv_yf(INSTRUMENT_US_PLTR_US_USD)
#DataDownloader.download_data_to_csv(INSTRUMENT_VCCY_ETH_USD)
DataDownloader.download_data_to_csv("USA500.IDX/USD")

ca = CandlestickAnalyzer()

dataframes = ca.execute()

results = []
for df in dataframes:
    results.append(run_backtest(Strategy_02, df))

for r in results:
    print(r)

