from trade_anayzer import TradeAnalyzer
from dukascopy_python.instruments import INSTRUMENT_US_PLTR_US_USD

ta = TradeAnalyzer()

ta.download_data(INSTRUMENT_US_PLTR_US_USD)

ta.create_tecnical_indicators()

ta.get_target()


input("Press Enter to exit...")