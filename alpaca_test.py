import alpaca_trade_api as tradeapi

# -----------------------------
# Configurazione API
# -----------------------------
API_KEY = "PK4TFTC9YYDHD5S51QNM".strip()
API_SECRET = "10KDzFIhvTokJUnMStGCW6IioaEUJaEdCoJch7qo".strip()
BASE_URL = "https://paper-api.alpaca.markets"  # paper trading

api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# -----------------------------
# Stato conto
# -----------------------------
account = api.get_account()
print("Account status:", account.status)
print("Cash available:", account.cash)
print("Portfolio value:", account.portfolio_value)

# -----------------------------
# Dati storici
# -----------------------------
symbol = "ETHUSD"
timeframe = "1H"  # 1h candles
start_date = "2025-08-01"
end_date = "2025-08-27"

#bars = api.get_bars(symbol, timeframe, start=start_date, end=end_date, feed='iex').df
#print()

# -----------------------------
# Esempio ordine BUY
# -----------------------------
# Acquista 0.01 ETH (paper trading)
order = api.submit_order(
    symbol=symbol,
    qty=1,
    side='buy',
    type='market',
    time_in_force='gtc'
)
print("Ordine inviato:", order)

# # -----------------------------
# # Esempio ordine SELL
# # -----------------------------
# # Vendi 0.01 ETH (paper trading)
# order = api.submit_order(
#     symbol=symbol,
#     qty=0.01,
#     side='sell',
#     type='market',
#     time_in_force='day'
# )
# print("Ordine inviato:", order)
