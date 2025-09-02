import pandas as pd
import talib
import numpy as np
from tqdm import tqdm
import itertools

class DataPreprocessor:
    def __init__(self, df):
        self.df_origin = df
        self.df_final = df.copy()

    def optimize_macd(self):        
        best_score = -np.inf
        best_params = None

        # Esempio di range dei periodi
        fast_range = np.arange(5, 50, 1)
        mid_range  = np.arange(10, 100, 1)
        slow_range = np.arange(20, 200, 1)

        total = len(fast_range) * len(mid_range) * len(slow_range)
        
        best_score = -1
        best_params = None
        pbar = tqdm(total=total, desc="Ottimizzazione MACD")

        for fast, slow, signal in tqdm(itertools.product(range(5,50), range(10,100), range(20,200))):
            if fast >= slow:  # condizione logica, fast deve essere < slow
                continue
            macd, signal_line, hist = talib.MACD(self.df_final["Close"], fastperiod=fast, slowperiod=slow, signalperiod=signal)
            
            # esempio di regola: quando hist > 0 siamo long, altrimenti flat
            self.df_final["pos"] = (hist > 0).astype(int)
            self.df_final["returns"] = self.df_final["Close"].pct_change() * self.df_final["pos"].shift()
            
            score = self.df_final["returns"].sum()  # qui puoi usare anche Sharpe ratio ecc.
            
            if score > best_score:
                best_score = score
                best_params = (fast, slow, signal)
            
            pbar.update(1)

        params = {"EMA_Fast_length": best_params[0], "EMA_Mid_length": best_params[1], "EMA_Slow_length": best_params[2]}
        
        print("Migliori parametri EMA (fast, mid, slow):", best_params, "Score:", best_score)
        
        exit()

    def genrate_dataset(self):

        macd_params = self.optimize_macd()

        macd, macdsignal, macdhist = talib.MACD(self.df_final["Close"], fastperiod=macd_params["EMA_Slow_length"], slowperiod=macd_params["EMA_Mid_length"], signalperiod=macd_params["EMA_Fast_length"])

        self.df_final["MACD"] = macd
        self.df_final["MACD_signal"] = macdsignal
        self.df_final["MACD_hist"] = macdhist

        self.df_final["MACD_Buy"] = (macd > macdsignal) & (macd.shift(1) <= macdsignal.shift(1))
        self.df_final["MACD_Sell"] = (macd < macdsignal) & (macd.shift(1) >= macdsignal.shift(1))

        # --- EMA ---
        self.df_final["EMA_5"] = talib.EMA(self.df_final["Close"], timeperiod=5)
        self.df_final["EMA_20"]  = talib.EMA(self.df_final["Close"], timeperiod=20)
        self.df_final["EMA_50"] = talib.EMA(self.df_final["Close"], timeperiod=50)
        self.df_final["EMA_100"] = talib.EMA(self.df_final["Close"], timeperiod=100)
        self.df_final["EMA_200"] = talib.EMA(self.df_final["Close"], timeperiod=200)

        # Normalizzazione rispetto al prezzo
        self.df_final["EMA_5_norm"] = self.df_final["EMA_5_norm"] / self.df_final["Close"]
        self.df_final["EMA_20_norm"]  = self.df_final["EMA_20_norm"]  / self.df_final["Close"]
        self.df_final["EMA_50_norm"] = self.df_final["EMA_50_norm"] / self.df_final["Close"]
        self.df_final["EMA_100_norm"]  = self.df_final["EMA_100_norm"]  / self.df_final["Close"]
        self.df_final["EMA_200_norm"] = self.df_final["EMA_200_norm"] / self.df_final["Close"]

        self.df_final["EMA_cross"] = (self.df_final["EMA_5_norm"] > self.df_final["EMA_50_norm"]).astype(int)
        self.df_final["EMA10" \
        "_slope_pct"] = self.df_final["EMA_10_norm"].pct_change()
        self.df_final["EMA20_slope_pct"] = self.df_final["EMA_20_norm"].pct_change()

        # --- RSI ---
        rsi = talib.RSI(self.df_final["Close"], timeperiod=14)
        self.df_final["RSI"] = rsi
        self.df_final["RSI_norm"] = rsi / 100
        self.df_final["RSI_Buy"] = rsi < 30
        self.df_final["RSI_Sell"] = rsi > 70

        # --- Bollinger Bands ---
        upper, middle, lower = talib.BBANDS(self.df_final["Close"], timeperiod=10, nbdevup=2.77, nbdevdn=2.5, matype=4)

        self.df_final["BB_Width"] = (upper - lower) / middle
        self.df_final["BB_Percent_b"] = (self.df_final["Close"] - lower) / (upper - lower)
        self.df_final["BB_Breakout_up"] = (self.df_final["Close"] > upper).astype(int)
        self.df_final["BB_Breakout_down"] = (self.df_final["Close"] < lower).astype(int)

        self.df_final["BB_Buy"] = (self.df_final["Close"] < lower).astype(int)
        self.df_final["BB_Sell"] = (self.df_final["Close"] > upper).astype(int)

        # Engulfing
        self.df_final["Engulfing_Buy"] = (self.df_final["Close"] > self.df_final["Open"]) & (self.df_final["Close"].shift(1) < self.df_final["Open"].shift(1))
        self.df_final["Engulfing_Sell"] = (self.df_final["Close"] < self.df_final["Open"]) & (self.df_final["Close"].shift(1) > self.df_final["Open"].shift(1))

        # OBV
        self.df_final["OBV"] = talib.OBV(self.df_final["Close"], self.df_final["Volume"])
        obv_signal = self.df_final["OBV"].diff() > 0
        self.df_final["OBV_Buy"] = obv_signal & (obv_signal.shift(1) == False)
        self.df_final["OBV_Sell"] = ~obv_signal & (obv_signal.shift(1) == True)

        # VWAP
        self.df_final["VWAP"] = (self.df_final["Close"] * self.df_final["Volume"]).cumsum() / self.df_final["Volume"].cumsum()
        self.df_final["VWAP_Buy"] = self.df_final["Close"] > self.df_final["VWAP"]
        self.df_final["VWAP_Sell"] = self.df_final["Close"] < self.df_final["VWAP"]

        # Money Flow Index (MFI)
        mfi = talib.MFI(self.df_final['High'], self.df_final['Low'], self.df_final['Close'], self.df_final['Volume'], timeperiod=14)
        self.df_final['MFI_Buy'] = mfi < 20
        self.df_final['MFI_Sell'] = mfi > 80

        # Volume Anomaly
        sma_volume = talib.SMA(self.df_final['Volume'], timeperiod=20)
        volume_spike = self.df_final['Volume'] > (sma_volume * 2.0)
        self.df_final['VolAnomaly_Buy'] = volume_spike & (self.df_final['Close'] > self.df_final['Open'])
        self.df_final['VolAnomaly_Sell'] = volume_spike & (self.df_final['Close'] < self.df_final['Open'])

        # -------------------
        # Breakout con conferma volumi + ADX
        # -------------------
        self.df_final["Donchian_High"] = self.df_final["High"].rolling(window=20).max()
        self.df_final["Donchian_Low"] = self.df_final["Low"].rolling(window=20).min()
        self.df_final["Vol_MA20"] = talib.SMA(self.df_final["Volume"], timeperiod=20)
        self.df_final["ADX"] = talib.ADX(self.df_final["High"], self.df_final["Low"], self.df_final["Close"], timeperiod=14)
        adx_threshold = 20

        self.df_final["Breakout_Up"] = (
            (self.df_final["Close"] > self.df_final["Donchian_High"].shift(1)) &
            (self.df_final["Volume"] > self.df_final["Vol_MA20"]) &
            (self.df_final["ADX"] > adx_threshold)
        )

        self.df_final["Breakout_Down"] = (
            (self.df_final["Close"] < self.df_final["Donchian_Low"].shift(1)) &
            (self.df_final["Volume"] > self.df_final["Vol_MA20"]) &
            (self.df_final["ADX"] > adx_threshold)
        )

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)

        print(self.df_final)

    def generate_targets(self, front_up_candles = 5, th_pct = 0.01):
        # Parametri
        N = front_up_candles # numero di candele future
        threshold_pct = th_pct  # soglia percentuale minima (es. 1%)

        # Calcolo ritorno futuro
        self.df_final['future_return'] = self.df_final['Close'].shift(-N) / self.df_final['Close'] - 1

        # Generazione target cautelativo
        def generate_cautious_target(r):
            if r >= threshold_pct:
                return 1    # rialzo significativo
            elif r <= -threshold_pct:
                return -1   # ribasso significativo
            else:
                return 0    # nessun movimento significativo

        self.df_final['Target'] = self.df_final['future_return'].apply(generate_cautious_target)

        # Controllo distribuzione target
        print(self.df_final['Target'].value_counts())
        print(self.df_final[['Close','future_return','Target']].tail(10))