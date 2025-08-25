from datetime import datetime, timedelta
import dukascopy_python
import pandas as pd
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_GBP_USD, INSTRUMENT_US_PLTR_US_USD
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--instrument", help="echo the string you use here", required=True)
args = parser.parse_args()
print("Strumento selezionato", args.instrument)


start = datetime(2017, 1, 1)
end = datetime(2024, 1, 1)
instrument = args.instrument
interval = dukascopy_python.INTERVAL_DAY_1
offer_side = dukascopy_python.OFFER_SIDE_BID

df = dukascopy_python.fetch(
    instrument,
    interval,
    offer_side,
    start,
    end,
)

df = df.reset_index()

print(df)

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

# Salva in CSV
df.to_csv(f"data/{instrument.split('.')[0]}.csv", index=False)

print(df)