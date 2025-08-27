from alpaca_trader import AlpacaTrader
import asyncio
from datetime import datetime, timedelta

async def main():
    at = AlpacaTrader("PK4TFTC9YYDHD5S51QNM", "10KDzFIhvTokJUnMStGCW6IioaEUJaEdCoJch7qo")
    await at.start_all()

asyncio.run(main())