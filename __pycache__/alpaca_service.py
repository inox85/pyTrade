

class AlpacaService:
    
    def __init__(self, api_key="PK4TFTC9YYDHD5S51QNM", api_secret="10KDzFIhvTokJUnMStGCW6IioaEUJaEdCoJch7qo", symbol="AAPL", usa_sync=True):
        self.base_url = "https://paper-api.alpaca.markets"
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.usa_sync = usa_sync
        self.account = self.api.get_account()
        self.positions = self.api.list_positions()
        self.orders = self.api.list_orders()

        print("Account status:", self.account.status)
        print("Positions:", self.positions)
        print("Portfolio value:", self.account.portfolio_value)
        print("Cash available:", self.account.cash)
