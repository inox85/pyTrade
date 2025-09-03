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
        fast_range = np.arange(5, 100, 1)
        mid_range  = np.arange(10, 200, 1)
        slow_range = np.arange(15, 300, 1)

        total = len(fast_range) * len(mid_range) * len(slow_range)
        
        best_score = -1
        best_params = None
        pbar = tqdm(total=total, desc="Ottimizzazione MACD")

        for fast, slow, signal in tqdm(itertools.product(range(5,100), range(10,200), range(15,300))):
            if fast >= slow:  # condizione logica, fast deve essere < slow
                pbar.update(1)
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

        self.df_final["MACD_Fast_length"] = best_params[0]
        self.df_final["MACD_Slow_length"] = best_params[1]
        self.df_final["MACD_Signal_length"] = best_params[2]

        print("Migliori parametri MACD (fast, slow, signal):", best_params, "Score:", best_score)
        

    def optimize_bbands(self):
        best_score = -np.inf
        best_params = None

        # Range dei parametri da ottimizzare
        timeperiod_range = np.arange(5, 30, 1)     # lunghezza media mobile
        nbdevup_range = np.arange(1, 4, 0.05)     # deviazione standard superiore
        nbdevdn_range = np.arange(1, 4, 0.05)     # deviazione standard inferiore

        total = len(timeperiod_range) * len(nbdevup_range) * len(nbdevdn_range)
        pbar = tqdm(total=total, desc="Ottimizzazione BBANDS")

        for tp, up, dn in itertools.product(timeperiod_range, nbdevup_range, nbdevdn_range):
            # Calcolo BBANDS
            upper, middle, lower = talib.BBANDS(self.df_final["Close"], timeperiod=tp, nbdevup=up, nbdevdn=dn, matype=0)

            # Semplice strategia: long se prezzo < lower, short se prezzo > upper
            pos = np.where(self.df_final["Close"] < lower, 1, np.where(self.df_final["Close"] > upper, -1, 0))

            # Calcolo dei ritorni giornalieri ponderati dalla posizione
            returns = self.df_final["Close"].pct_change().shift(-1) * pos  # shift(-1) per ritorno futuro

            # Score = somma dei ritorni cumulati (puoi sostituire con Sharpe ratio)
            score = np.nansum(returns)

            if score > best_score:
                best_score = score
                best_params = (tp, up, dn)

            pbar.update(1)

        pbar.close()
        params = {"timeperiod": best_params[0], "nbdevup": best_params[1], "nbdevdn": best_params[2]}
        
        print("Migliori parametri BBANDS:", best_params, "Score:", best_score)
        return params, best_score

    def genrate_dataset(self):

        #macd_params = self.optimize_macd()

        #macd, macdsignal, macdhist = talib.MACD(self.df_final["Close"], fastperiod=macd_params["EMA_Slow_length"], slowperiod=macd_params["EMA_Mid_length"], signalperiod=macd_params["EMA_Fast_length"])
        macd, macdsignal, macdhist = talib.MACD(self.df_final["Close"], fastperiod=5, slowperiod=14, signalperiod=28)

        self.df_final["MACD"] = macd
        self.df_final["MACD_signal"] = macdsignal
        self.df_final["MACD_hist"] = macdhist

        self.df_final["MACD_Buy"] = (macd > macdsignal) & (macd.shift(1) <= macdsignal.shift(1))
        self.df_final["MACD_Sell"] = (macd < macdsignal) & (macd.shift(1) >= macdsignal.shift(1))

        periods = [5, 10, 20, 50, 100]

        # =========================================
        # 3) Calcolo SMA e EMA
        # =========================================
        for p in periods:
            self.df_final[f"SMA_{p}"] = self.df_final["Close"].rolling(window=p).mean()
            self.df_final[f"EMA_{p}"] = self.df_final["Close"].ewm(span=p, adjust=False).mean()

        # =========================================
        # 4) Calcolo slope percentuale
        # =========================================
        window_slope = 5  # numero di periodi per calcolare la pendenza

        # slope percentuale sul prezzo
        self.df_final["Close_slope_pct"] = (self.df_final["Close"] - self.df_final["Close"].shift(window_slope)) / self.df_final["Close"].shift(window_slope)

        # slope percentuale su tutte le medie
        ma_columns = [col for col in self.df_final.columns if col.startswith(("SMA", "EMA"))]
        for col in ma_columns:
            self.df_final[f"{col}_slope_pct"] = (self.df_final[col] - self.df_final[col].shift(window_slope)) / self.df_final[col].shift(window_slope)

        # =========================================
        # 5) Funzioni per incroci
        # =========================================
        def crossover(series1, series2):
            return (series1 > series2) & (series1.shift(1) <= series2.shift(1))

        def crossunder(series1, series2):
            return (series1 < series2) & (series1.shift(1) >= series2.shift(1))

        # =========================================
        # 6) Genera incroci per tutte le coppie di medie
        # =========================================
        new_cols = {}
        for i in range(len(ma_columns)):
            for j in range(i+1, len(ma_columns)):
                col1, col2 = ma_columns[i], ma_columns[j]
                new_cols[f"{col1}_x_{col2}_UP"] = crossover(self.df_final[col1], self.df_final[col2])
                new_cols[f"{col1}_x_{col2}_DOWN"] = crossunder(self.df_final[col1], self.df_final[col2])

        # Concateno tutto insieme in una volta
        self.df_final = pd.concat([self.df_final, pd.DataFrame(new_cols, index=self.df_final.index)], axis=1)

        # --- RSI ---
        rsi = talib.RSI(self.df_final["Close"], timeperiod=14)
        self.df_final["RSI"] = rsi
        self.df_final["RSI_norm"] = rsi / 100
        self.df_final["RSI_Buy"] = rsi < 30
        self.df_final["RSI_Sell"] = rsi > 70

        self.optimize_bbands()
        exit()
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


    def generate_targets(self, front_up_candles = 5, th_pct = 0.01):
        # Parametri
        N = front_up_candles
        threshold_pct = th_pct

        # Calcolo ritorno futuro
        self.df_final['future_return'] = self.df_final['Close'].shift(-N) / self.df_final['Close'] - 1

        # Generazione target cautelativo
        def generate_cautious_target(r):
            if r >= threshold_pct:
                return 1    # rialzo significativo
            elif r <= -threshold_pct:
                return -1   # ribasso significativo
            else:
                return 0

        self.df_final['Target'] = self.df_final['future_return'].apply(generate_cautious_target)

        # Calcolo profitto percentuale basato sul target
        self.df_final['Profit_pct'] = np.where(
            self.df_final['Target'] == 1,
            (self.df_final['Close'].shift(-N) - self.df_final['Close']) / self.df_final['Close'],  # long
            np.where(
                self.df_final['Target'] == -1,
                (self.df_final['Close'] - self.df_final['Close'].shift(-N)) / self.df_final['Close'],  # short
                0
            )
        )

        # Stampiamo DataFrame completo
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        print(self.df_final)

        # Profitto percentuale totale (somma dei singoli trade)
        total_profit_pct = self.df_final['Profit_pct'].sum()
        print(f"\nProfitto teorico totale (%) cumulato: {total_profit_pct*100:.2f}%")
