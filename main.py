from trade_anayzer import TradeAnalyzer
from dukascopy_python.instruments import INSTRUMENT_US_PLTR_US_USD

ta = TradeAnalyzer()

ta.load_data(INSTRUMENT_US_PLTR_US_USD)

#ta.load_data("USA500.IDX/USD")

#ta.create_tecnical_indicators()

#ta.get_target()

#ta.train_model()

ta.create_normalized_tecnical_indicators()

ta.get_target()

ta.train_model()

input("Press Enter to exit...")