import pandas as pd
import talib
import numpy as np
from tqdm import tqdm
import itertools
import json
import os

class DataPreprocessor:
    def __init__(self, df, symbol, interval, recalculate_params=False):      
        self.interval = interval
        self.df_final = df.copy()
        self.symbol = symbol
        self.params_file = "params/params.json"
        data = self.load_json_safe(self.params_file, True)
        self.recalculate_params = recalculate_params

        if self.symbol not in data:
            print("Parametri non trovati per il simbolo:", self.symbol)
            print("Il campo verrà creato e verranno calcolati i parametri ottimizati")
            
            data[self.symbol] = {}
            with open(self.params_file, "w") as f:
                json.dump(data, f, indent=4)
            self.recalculate_params = True

        if self.interval not in data[self.symbol]:
            print("Parametri non trovati per il simbolo:", self.interval)
            print("Il campo verrà creato e verranno calcolati i parametri ottimizati")

            data[self.symbol][self.interval] = {}
            with open(self.params_file, "w") as f:
                json.dump(data, f, indent=4)
            self.recalculate_params = True

        self.config_params = data[self.symbol][self.interval]

        required_keys = ["obv", "volumeAdx", "mfi", "rsi", "tradeCountNorm", "bBands", "macd"]

        missing_keys = required_keys - self.config_params.keys()

        if missing_keys:
            print(f"Mancano queste chiavi: {missing_keys}")
            self.recalculate_params = True
        else:
            print(f"Tutte le chiavi sono presenti per {self.symbol} {self.interval}✅")
            self.recalculate_params = False

        if self.recalculate_params:
            self.calculate_params(missing_keys)
        else:
            self.load_params()

        print(self.config_params)


    def calculate_params(self, missing_params = []):
        print("Inizio ricalcolo parametri...")
        if "macd" in missing_params:
            self.macd_params = self.optimize_macd()
        if "obv" in missing_params: 
            self.obv_params = self.optimize_obv()
        if "volumeAdx" in missing_params:
            self.volume_params = self.optimize_volume_adx()
        if "mfi" in missing_params:
            self.mfi_params = self.optimize_mfi()
        if "rsi" in missing_params:   
            self.rsi_params = self.optimize_rsi()
        if "tradeCountNorm" in missing_params:
            self.trade_count_params =self.optimize_trade_count_norm()
        if "bBands" in missing_params:
            self.bb_params = self.optimize_bbands()

    def load_params(self):
        print("Inizio caricamento parametri...")
        self.obv_params = self.config_params["obv"]
        self.volume_params = self.config_params["volumeAdx"]
        self.macd_params = self.config_params["macd"] 
        self.mfi_params = self.config_params["mfi"]  
        self.rsi_params = self.config_params["rsi"]
        self.trade_count_params = self.config_params["tradeCountNorm"]
        self.bb_params = self.config_params["bBands"]

    def load_json_safe(self, filepath, init_empty=True):
        """
        Carica un file JSON in modo sicuro.
        - Se non esiste → restituisce {} (e lo crea se init_empty=True)
        - Se è vuoto o malformato → restituisce {} (e lo riscrive se init_empty=True)
        - Altrimenti restituisce i dati caricati
        """
        data = {}

        if not os.path.exists(filepath):
            if init_empty:
                with open(filepath, "w") as f:
                    json.dump({}, f, indent=4)
            return data

        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # File esistente ma vuoto o malformato
                data = {}
                if init_empty:
                    with open(filepath, "w") as fw:
                        json.dump({}, fw, indent=4)

        return data
    
    def save_params(self, field, params):
        with open(self.params_file, "r") as f:
            data = json.load(f)
            data[self.symbol][self.interval][field] = params
        
        with open(self.params_file, "w") as f:
            json.dump(data, f, indent=4)

    def get_score(self, close, pos, horizons=[1,3,5],metric="cumulative"):
        horizon_scores = []
        for h in horizons:
            future_returns = close.pct_change(h).shift(-h)
            strat_returns = future_returns * pos

            if metric == "cumulative":
                score_h = strat_returns.sum() * 100
            elif metric == "sharpe":
                mean_ret = np.nanmean(strat_returns)
                std_ret = np.nanstd(strat_returns)
                score_h = mean_ret / std_ret if std_ret != 0 else -np.inf
            elif metric == "accuracy":
                correct_signals = np.sum((strat_returns > 0) & (pos != 0))
                total_signals = np.sum(pos != 0)
                score_h = correct_signals / total_signals if total_signals > 0 else 0
            else:
                raise ValueError("Metric deve essere 'cumulative', 'sharpe' o 'accuracy'")

            horizon_scores.append(score_h)

        score = sum(horizon_scores)
        return score


    def optimize_macd(self, 
                  fast_range=(5, 20), 
                  slow_range=(26, 100), 
                  signal_range=(5, 20), 
                  horizons=[1, 3, 5],
                  weights=None,
                  metric="cumulative"):
        """
        Ottimizza i parametri del MACD con una grid search su più orizzonti temporali.

        Parametri:
        ----------
        horizons : list[int]
            Lista di orizzonti futuri (in numero di candele).
        weights : list[float] or None
            Pesi per ogni orizzonte (somma = 1). Se None -> pesi uniformi.
        metric : str
            "cumulative" -> somma dei ritorni
            "sharpe" -> rapporto ritorno/std
            "accuracy" -> % segnali corretti

        Ritorna:
        --------
        dict con parametri migliori e score.
        """
        import numpy as np
        import itertools
        import talib
        from tqdm import tqdm

        if weights is None:
            weights = [1/len(horizons)] * len(horizons)

        best_score = -np.inf
        best_params = None

        close = self.df_final["Close"]

        # Creiamo la griglia come lista, così può essere riutilizzata
        grid = list(itertools.product(
            range(fast_range[0], fast_range[1]),
            range(slow_range[0], slow_range[1]),
            range(signal_range[0], signal_range[1])
        ))

        total = len(grid) # due volte per invert=False e True

        for invert in [False, True]:
            pbar = tqdm(total=total, desc=f"Ottimizzazione MACD multi-horizon Inversione->{invert}")
            for fast, slow, signal in grid:
                if fast >= slow:
                    pbar.update(1)
                    continue

                macd, signal_line, hist = talib.MACD(
                    close,
                    fastperiod=fast,
                    slowperiod=slow,
                    signalperiod=signal
                )

                # Segnali -1/0/1 e inversione
                if invert:
                    pos = np.where(hist > 0, -1, np.where(hist < 0, 1, 0))
                else:
                    pos = np.where(hist > 0, 1, np.where(hist < 0, -1, 0))

                num_signals = np.sum(pos != 0)
                if num_signals == 0:
                    pbar.update(1)
                    continue

                score = self.get_score(close, pos)

                if score > best_score:
                    best_score = score
                    best_params = (fast, slow, signal, invert, num_signals)
                # Aggiornamento progress bar
                postfix_str = (f"best_score={best_score:.5f} signals={num_signals}")
                pbar.set_postfix_str(postfix_str)
                pbar.update(1)
            pbar.close()

            params = { 
                "MACD_Fast_length": best_params[0],
                "MACD_Slow_length": best_params[1],
                "MACD_Signal_length": best_params[2],
                "Best_Score": round(float(best_score), 6),
                "Metric": metric,
                "Horizons": horizons,
                "Inverted": best_params[3],
                "Signals":int(best_params[4]), 
            }
            print(params)

        self.save_params("macd", params)

        return params

    def optimize_rsi(self, horizons=[1,3,5],
                 rsi_period_range=range(5, 31, 1),
                 rsi_low_range=range(20, 41, 1),
                 rsi_high_range=range(60, 81, 1),
                 metric="cumulative"):
        """
        Ottimizza RSI (timeperiod + soglie low/high) su più orizzonti temporali.

        Parametri:
            horizons: lista di int -> orizzonti temporali in candele
            rsi_period_range: range di valori per il timeperiod RSI
            rsi_low_range: range per la soglia inferiore (oversold)
            rsi_high_range: range per la soglia superiore (overbought)
            metric: str -> "cumulative", "sharpe", "accuracy"

        Ritorna:
            dict con i migliori parametri e score
        """
        import numpy as np
        import talib
        import itertools
        from tqdm import tqdm

        best_score = -np.inf
        best_params = None

        close = self.df_final["Close"]

        total = len(rsi_period_range) * len(rsi_low_range) * len(rsi_high_range)

        for invert in [False, True]:                 
            pbar = tqdm(total=total, desc=f"Ottimizzazione RSI Inversione->{invert}")
            for period, low_thr, high_thr in itertools.product(rsi_period_range,
                                                            rsi_low_range,
                                                            rsi_high_range):
                if low_thr >= high_thr:  # soglie non valide
                    continue

                # --- RSI ---
                rsi = talib.RSI(close, timeperiod=period)

                # --- Segnali ---
                if invert:
                    buy_signal = rsi > low_thr
                    sell_signal = rsi < high_thr
                else:
                    buy_signal = rsi < low_thr
                    sell_signal = rsi > high_thr

                pos = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))

                num_signals = np.sum(pos != 0)
                if num_signals == 0:
                    pbar.update(1)
                    continue

                score = self.get_score(close, pos)

                if score > best_score:
                    best_score = score
                    best_params = (period, low_thr, high_thr, invert, num_signals)

                                # Aggiornamento progress bar
                postfix_str = (f"best_score={best_score:.5f} signals={num_signals}")
                pbar.set_postfix_str(postfix_str)
                pbar.update(1)     
            pbar.close()


            params = {
                "Best_RSI_Period": best_params[0],
                "Best_RSI_Low": best_params[1],
                "Best_RSI_High": best_params[2],
                "Best_Score": round(float(best_score),6),
                "Metric": metric,
                "Horizons": horizons,
                "Inverted": best_params[3],
                "Signals": int(best_params[4])
            }
            print("Ottimizzazione parametri RSI:", params)

        self.save_params("rsi", params)

        
        return params

    def optimize_mfi(self, horizons=[1, 3, 5],
                 timeperiod_range=range(5, 50),
                 low_thr_range=range(10, 41, 1),
                 high_thr_range=range(60, 91, 1),
                 metric="cumulative"):
        """
        Ottimizza i parametri del MFI (Money Flow Index).
        """
        import numpy as np
        import talib
        import itertools
        from tqdm import tqdm

        best_score = -np.inf
        best_params = None

        close = self.df_final["Close"]
        high = self.df_final["High"]
        low = self.df_final["Low"]
        volume = self.df_final["Volume"]

        total = len(timeperiod_range) * len(low_thr_range) * len(high_thr_range)

        for invert in [False, True]:
            pbar = tqdm(total=total, desc=f"Ottimizzazione MFI Inversione->{invert}")
            for tp, low_thr, high_thr in itertools.product(timeperiod_range, low_thr_range, high_thr_range):
                if low_thr >= high_thr:  # soglie non valide
                    continue

                # Calcolo MFI
                mfi = talib.MFI(high, low, close, volume, timeperiod=tp)

                # Strategia base: long se MFI < low_thr, short se MFI > high_thr
                base_pos = np.where(mfi < low_thr, 1, np.where(mfi > high_thr, -1, 0))

                # Applica inversione se richiesto
                pos = -base_pos if invert else base_pos
            
                num_signals = np.sum(pos != 0)
                if num_signals == 0:
                    pbar.update(1)
                    continue

                score = self.get_score(close, pos)

                if score > best_score:
                    best_score = score
                    best_params = (tp, low_thr, high_thr, invert)

                postfix_str = (f"best_score={best_score:.5f} signals={num_signals}")
                pbar.set_postfix_str(postfix_str)
                pbar.update(1)
            pbar.close()


            params = {
                "MFI_TimePeriod": int(best_params[0]),
                "MFI_Low": int(best_params[1]),
                "MFI_High": int(best_params[2]),
                "Best_Score": round(float(best_score), 6),
                "Metric": metric,
                "Horizons": horizons,
                "Inverted": best_params[3],
            }
            print(params)

        self.save_params("mfi", params)
        
        return params
    
    def optimize_obv(self, horizons=[1,3,5],
                 slope_window_range=range(3, 11),
                 momentum_window_range=range(10, 51, 1),
                 metric="cumulative"):
        """
        Ottimizza le due lunghezze rolling di OBV:
            - slope_window: finestra per rolling mean della derivata OBV
            - momentum_window: finestra per media mobile OBV (per momentum)
        
        Parametri:
            horizons: lista di int -> orizzonti temporali in candele
            slope_window_range: range di valori per la finestra della derivata OBV
            momentum_window_range: range per la finestra della media OBV
            metric: str -> "cumulative", "sharpe", "accuracy"

        Ritorna:
            dict con i migliori parametri e score
        """
        import numpy as np
        import talib
        import itertools
        from tqdm import tqdm

        best_score = -np.inf
        best_params = None

        close = self.df_final["Close"]
        volume = self.df_final["Volume"]

        total = len(slope_window_range) * len(momentum_window_range)

        for invert in [False, True]:
            pbar = tqdm(total=total, desc=f"Ottimizzazione OBV Inversione-{invert}")
            for slope_win, mom_win in itertools.product(slope_window_range, momentum_window_range):
                # --- Calcolo OBV ---
                obv = talib.OBV(close, volume)
                obv_diff = obv.diff()
                obv_slope = obv_diff.rolling(slope_win).mean()

                # Segnale numerico semplice: 1 se slope positivo, -1 se negativo
                pos = np.where(obv_slope > 0, 1, np.where(obv_slope < 0, -1, 0))

                if invert:
                    pos = -pos

                num_signals = np.sum(pos != 0)
                if num_signals == 0:
                    pbar.update(1)
                    continue

                # Calcolo score (puoi usare la tua funzione get_score)
                score = self.get_score(close, pos)

                if score > best_score:
                    best_score = score
                    best_params = (slope_win, mom_win, invert, num_signals)

                # Aggiornamento progress bar
                postfix_str = (f"best_score={best_score:.5f} signals={num_signals}")
                pbar.set_postfix_str(postfix_str)
                pbar.update(1)

            pbar.close()

            params = {
                "Best_OBV_Slope_Window": best_params[0],
                "Best_OBV_Momentum_Window": best_params[1],
                "Best_Score": round(float(best_score),6),
                "Metric": metric,
                "Horizons": horizons,
                "Signals": int(best_params[3]),
                "Inverted:": best_params[2]       
            }
            print(params)

        self.save_params("obv", params)

        return params

    def optimize_bbands(self, horizons=[1, 3, 5], metric="cumulative"):
        """
        Ottimizza i parametri delle Bollinger Bands (BBANDS).

        Parametri:
        ----------
        horizons : list[int]
            Lista di orizzonti futuri (in numero di candele).
        metric : str
            "cumulative" -> somma dei ritorni
            "sharpe" -> rapporto ritorno/std
            "accuracy" -> % segnali corretti
        """
        import numpy as np
        import talib
        import itertools
        from tqdm import tqdm

        best_score = -np.inf
        best_params = None

        timeperiod_range = np.arange(5, 30, 1)
        nbdevup_range = np.arange(1, 4, 0.05)
        nbdevdn_range = np.arange(1, 4, 0.05)
        matype_range = [0, 1, 2, 3, 4]  # limito i tipi di MA per performance

        total = len(timeperiod_range) * len(nbdevup_range) * len(nbdevdn_range) * len(matype_range)
        

        close = self.df_final["Close"]
        for invert in  [False, True]:
            pbar = tqdm(total=total, desc="Ottimizzazione BBANDS")
            for tp, up, dn, matype in itertools.product(timeperiod_range, nbdevup_range, nbdevdn_range, matype_range):

                upper, middle, lower = talib.BBANDS(close, timeperiod=tp, nbdevup=up, nbdevdn=dn, matype=matype)

                pos = np.where(close < lower, 1, np.where(close > upper, -1, 0))
                
                if invert:
                    pos = -pos

                num_signals = np.sum(pos != 0)
                if num_signals == 0:
                    pbar.update(1)
                    continue

                score = self.get_score(close, pos)

                if score > best_score:
                    best_score = score
                    best_params = (tp, up, dn, matype, invert, num_signals)
                    
                postfix_str = (f"best_score={best_score:.5f} signals={num_signals}")
                pbar.set_postfix_str(postfix_str)

                pbar.update(1)

            pbar.close()


            params = {
                "BBANDS_TimePeriod": int(best_params[0]),
                "BBANDS_NBDevUp": round(float(best_params[1]), 2),
                "BBANDS_NBDevDn": round(float(best_params[2]), 2),
                "BBANDS_MAType": int(best_params[3]),
                "Best_Score": round(float(best_score), 6),
                "Metric": metric,
                "Horizons": horizons,
                "Inverted:": best_params[4],
                "Signals":int(best_params[5])
            }
            print(params)

        self.save_params("bBands", params)
        
        return params


# --------------------------
# Esempio: dati OHLCV
# df deve avere colonne: ['Open', 'High', 'Low', 'Close', 'Volume']

# Funzione di ottimizzazione
# --------------------------
    def optimize_volume_adx(self, 
                        horizons=[1,3,5],
                        donchian_range=np.arange(2,16,1),
                        vol_ma_range=np.arange(2,16,0.5),
                        adx_period_range=np.arange(5,15,1),
                        adx_threshold_range=np.arange(5,16,0.5),
                        vol_mult_range=np.arange(1.05,1.5,0.5),
                        metric="cumulative"):
        """
        Ottimizza Volume Spike + ADX + Donchian su più orizzonti temporali.
        """

        best_score = -np.inf
        best_params = None

        close = self.df_final["Close"]
        open_ = self.df_final["Open"]
        high = self.df_final["High"]
        low = self.df_final["Low"]
        volume = self.df_final["Volume"]

        total = (len(donchian_range) * len(vol_ma_range) * len(adx_period_range) *
                len(adx_threshold_range) * len(vol_mult_range))
        

        for invert in [False, True]:
            pbar = tqdm(total=total, desc=f"Ottimizzazione Volume+ADX Inversione->{invert}")
            for donchian_w, vol_ma, adx_period, adx_thr, vm in itertools.product(
                    donchian_range, vol_ma_range, adx_period_range, adx_threshold_range, vol_mult_range):

                # --- Indicatori ---
                Vol_MA = talib.SMA(volume, timeperiod=vol_ma)
                ADX_val = talib.ADX(high, low, close, timeperiod=adx_period)
                donchian_high = high.shift(1).rolling(window=donchian_w).max()
                donchian_low = low.shift(1).rolling(window=donchian_w).min()

                # --- Segnali ---
                volume_spike = volume > (Vol_MA * vm)
                breakout_up = close > donchian_high
                breakout_down = close < donchian_low


                buy_signal = volume_spike & breakout_up & (ADX_val > adx_thr)
                sell_signal = volume_spike & breakout_down & (ADX_val > adx_thr)

                pos = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))

                # --- Inversione completa ---
                if invert:
                    pos = -pos

                num_signals = np.sum(pos != 0)
                if num_signals == 0:
                    pbar.update(1)
                    continue

                score = self.get_score(close, pos)

                if score > best_score:
                    best_score = score
                    best_params = (donchian_w, vol_ma, adx_period, adx_thr, vm, invert, num_signals)

                # Aggiornamento progress bar
                postfix_str = (f"best_score={best_score:.5f} signals={num_signals}")
                pbar.set_postfix_str(postfix_str)
                pbar.update(1)

            pbar.close()


            params = {
                "Best_Donchian_Window": int(best_params[0]),
                "Best_Vol_MA": int(best_params[1]),
                "Best_ADX_Timeperiod": int(best_params[2]),
                "Best_ADX_Threshold": round(float(best_params[3]),2),
                "Best_Volume_Multiplier": round(float(best_params[4]),2),
                "Best_Score": float(best_score),
                "Metric": metric,
                "Horizons": horizons,
                "Inverted": best_params[5],
                "Signals":int(best_params[6])
            }
            print(params)

        self.save_params("volumeAdx", params)
       
        return params
 
    def optimize_trade_count_norm(self, horizons=[1,3,5],
                              window_range=range(2, 100),
                              metric="cumulative",
                              invert_strategy=[False, True]):
        """
        Ottimizza la lunghezza della media mobile per trade_count_norm su più orizzonti temporali.

        Parametri:
            horizons: lista di int -> orizzonti temporali in candele
            window_range: range dei valori della finestra rolling
            metric: str -> "cumulative", "sharpe", "accuracy"

        Ritorna:
            dict con la finestra ottimale e score
        """
        import numpy as np
        from tqdm import tqdm

        best_score = -np.inf
        best_window = None

        close = self.df_final["Close"]
        trade_count = self.df_final["trade_count"]

        total = len(window_range)

        for invert in invert_strategy:
            pbar = tqdm(total=total, desc=f"Ottimizzazione trade_count_norm Inversione->{invert}")
            for window in window_range:
                # --- Calcolo trade_count_norm ---
                trade_count_norm = trade_count / trade_count.rolling(window).mean()

                # Segnali basati su trade_count_norm > 1 => attività alta, < 1 => bassa
                # if invert:
                #     buy_signal = trade_count_norm < 1
                #     sell_signal = trade_count_norm > 1
                # else:
                buy_signal = trade_count_norm > 1
                sell_signal = trade_count_norm < 1

                pos = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))

                if invert:
                    pos = -pos

                num_signals = np.sum(pos != 0)
                if num_signals == 0:
                    pbar.update(1)
                    continue

                score = self.get_score(close, pos)

                if score > best_score:
                    best_score = score
                    best_params = (window, invert, num_signals)

                pbar.update(1)  
            pbar.close()

            params = {
                "Best_TradeCountNorm_Window": best_params[0],
                "Best_Score": round(float(best_score), 6),
                "Metric": metric,
                "Horizons": horizons,
                "Inverted": best_params[1],
                "Signals": int(best_params[2])
            }
            print(params)
        
        self.save_params("tradeCountNorm", params)
        return params

    def get_dataset(self):
        return self.df_final

    def generate_tecnical_dataset(self, df):
        print("Carico parametri da json...")

        self.load_params()

        print("Inizio elaborazione dataset...")

        macd, macdsignal, macdhist = talib.MACD(df["Close"], fastperiod=self.macd_params["MACD_Fast_length"], slowperiod=self.macd_params["MACD_Slow_length"], signalperiod=self.macd_params["MACD_Signal_length"])

        df["MACD"] = macd
        df["MACD_signal "] = macdsignal
        df["MACD_hist"] = macdhist

        df["MACD_diff"] = macd - macdsignal
        df["MACD_slope"] = macd.diff()
        df["Signal_slope"] = macdsignal.diff()

        # df["MACD_Buy"] = (macd > macdsignal) & (macd.shift(1) <= macdsignal.shift(1))
        # df["MACD_Sell"] = (macd < macdsignal) & (macd.shift(1) >= macdsignal.shift(1))

        periods = [5, 10, 20, 50, 100]

        # =========================================
        # 3) Calcolo SMA e EMA
        # =========================================
        for p in periods:
            #df[f"SMA_{p}"] = df["Close"].rolling(window=p).mean()
            df[f"EMA_{p}"] = df["Close"].ewm(span=p, adjust=False).mean()
            df[f"MACD_Hist_volatility_{p}"] = macdhist.rolling(p).std()

        # =========================================
        # 4) Calcolo slope percentuale
        # =========================================

        window_slope = 5  # numero di periodi per calcolare la pendenza

        # slope percentuale sul prezzo
        df["Close_slope_pct"] = (df["Close"] - df["Close"].shift(window_slope)) / df["Close"].shift(window_slope)

        # slope percentuale su tutte le medie
        ma_columns = [col for col in df.columns if col.startswith(("SMA", "EMA"))]
        for col in ma_columns:
            df[f"{col}_slope_pct"] = (df[col] - df[col].shift(window_slope)) / df[col].shift(window_slope)

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
            for j in range(i + 1, len(ma_columns)):
                col1, col2 = ma_columns[i], ma_columns[j]

                # distanza percentuale tra le medie
                df[f"{col1}_x_{col2}_diff"] = (df[col1] - df[col2]) / df[col2]

                # forza dell'incrocio (solo se oggi c'è crossover/crossunder)
                df[f"{col1}_x_{col2}_UP_cont"] = df[f"{col1}_x_{col2}_diff"].where(crossover(df[col1], df[col2]), 0)
                df[f"{col1}_x_{col2}_DOWN_cont"] = df[f"{col1}_x_{col2}_diff"].where(crossunder(df[col1], df[col2]), 0)

        # Concateno tutto insieme in una volta
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # --- RSI ---
        rsi = talib.RSI(df["Close"], timeperiod=self.rsi_params["Best_RSI_Period"])
        df["RSI"] = rsi
        df["RSI_norm"] = rsi / 100
        #df["RSI_Buy"] = rsi < 30
        #df["RSI_Sell"] = rsi > 70

        # --- Bollinger Bands ---
        upper, middle, lower = talib.BBANDS(df["Close"], timeperiod=self.bb_params["BBANDS_TimePeriod"], nbdevup=self.bb_params["BBANDS_NBDevUp"], nbdevdn=self.bb_params["BBANDS_NBDevDn"], matype=self.bb_params["BBANDS_MAType"])

        # Larghezza BB
        df["BB_Width"] = (upper - lower) / middle

        # Percentuale posizione prezzo tra le bande
        df["BB_Percent_b"] = (df["Close"] - lower) / (upper - lower)

        # Breakout continuo: distanza dal bordo superiore/inferiore
        df["BB_Breakout_up_cont"] = df["Close"] - upper  # >0 indica quanto è sopra l'upper band
        df["BB_Breakout_down_cont"] = lower - df["Close"]  # >0 indica quanto è sotto la lower band

        df["Body_today"] = abs(df["Close"] - df["Open"])
        df["Body_prev"] = abs(df["Close"].shift(1) - df["Open"].shift(1))

        # Rapporto dimensioni
        df["Engulfing_ratio"] = df["Body_today"] / (df["Body_prev"] + 1e-9)

        # Direzione candela attuale (+1 verde, -1 rossa, 0 doji)
        df["Engulfing_direction"] = np.sign(df["Close"] - df["Open"])

        # Forza rispetto all'apertura precedente
        df["Engulfing_strength"] = (
            (df["Close"] - df["Open"].shift(1)) * df["Engulfing_direction"]
        )

        # 🔹 Calcolo dei corpi (open/close, no shadow)
        prev_low_body  = df[["Open", "Close"]].shift(1).min(axis=1)
        prev_high_body = df[["Open", "Close"]].shift(1).max(axis=1)
        today_low_body  = df[["Open", "Close"]].min(axis=1)
        today_high_body = df[["Open", "Close"]].max(axis=1)

        # Fattore di inclusione (quanto del corpo precedente è contenuto nell'attuale)
        df["Engulfing_inclusion"] = (
            (np.minimum(today_high_body, prev_high_body) - np.maximum(today_low_body, prev_low_body))
            / (prev_high_body - prev_low_body + 1e-9)
        )

        # 🔹 Quantificazione finale = forza * inclusione
        df["Engulfing_index"] = (
            df["Engulfing_ratio"] *
            df["Engulfing_strength"] *
            df["Engulfing_inclusion"]
        )
        # Engulfing
        #df["Engulfing_Buy"] = (df["Close"] > df["Open"]) & (df["Close"].shift(1) < df["Open"].shift(1))
        #df["Engulfing_Sell"] = (df["Close"] < df["Open"]) & (df["Close"].shift(1) > df["Open"].shift(1))

        # OBV
        df["OBV"] = talib.OBV(df["Close"], df["Volume"])

        # Differenze e momentum continui
        df["OBV_diff"] = df["OBV"].diff()
        df["OBV_pct"] = df["OBV"].pct_change()
        df["OBV_slope"] = df["OBV"].diff().rolling(self.obv_params["Best_OBV_Slope_Window"]).mean()
        df["OBV_momentum"] = df["OBV"] - df["OBV"].rolling(self.obv_params["Best_OBV_Momentum_Window"]).mean()

        # VWAP
        df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
        df["Close_VWAP_Diff"] = df["Close"] - df["VWAP"]
        df["Close_VWAP_Ratio"] = df["Close"] / df["VWAP"]

        df["Avg_Trade_Size"] = df["Volume"] / df["trade_count"]

        for l in [3, 5, 10]:
            df[f"Trade_Count_Norm_{l}"] = df["trade_count"] / df["trade_count"].rolling(l).mean()

        # Money Flow Index (MFI)

        mfi = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=self.mfi_params["MFI_TimePeriod"])

        # Volume Anomaly

        sma_volume = talib.SMA(df['Volume'], timeperiod=self.volume_params["Best_Vol_MA"])
        volume_spike = df['Volume'] > (sma_volume * self.volume_params["Best_Volume_Multiplier"])

        # -------------------
        # Breakout con conferma volumi + ADX
        # -------------------
        df["Donchian_High"] = df["High"].rolling(window=self.volume_params["Best_Donchian_Window"]).max()
        df["Donchian_Low"] = df["Low"].rolling(window=self.volume_params["Best_Donchian_Window"]).min()
        df["Vol_MA"] = talib.SMA(df["Volume"], timeperiod=self.volume_params["Best_Vol_MA"])
        df["ADX"] = talib.ADX(df["High"], df["Low"], df["Close"], timeperiod=self.volume_params["Best_ADX_Timeperiod"])
        adx_threshold = self.volume_params["Best_ADX_Threshold"]

        # Distanza del prezzo dalla banda superiore/inferiore
        df["Donchian_Close_HighDiff"] = df["Close"] - df["Donchian_High"]
        df["Donchian_Close_LowDiff"] = df["Close"] - df["Donchian_Low"]

        # Percentuale relativa
        df["Donchian_Close_HighPct"] = (df["Close"] - df["Donchian_High"]) / df["Donchian_High"]
        df["Donchian_Close_LowPct"] = (df["Close"] - df["Donchian_Low"]) / df["Donchian_Low"]

        df["Vol_Ratio"] = df["Volume"] / (df["Vol_MA"] + 1e-9)  # rapporto tra volume attuale e media
        df["Vol_Diff"] = df["Volume"] - df["Vol_MA"]

        df["ADX_Above_Threshold"] = df["ADX"] - adx_threshold  # >0 trend forte, <0 trend debole
        df["ADX_Slope"] = df["ADX"].diff()  # pendenza ADX

        print(df.columns)

        return df

    def generate_targets_dataset(self, df, horizons=[1, 5, 10, 20, 50], thresholds=None):
        """
        Genera target multi-orizzonte coerenti:
        - Target_{h}: Buy (1), Sell (-1), Hold (0) basati su soglie percentuali
        - Profit_{h}: rendimento percentuale nei prossimi h passi

        :param df: DataFrame con almeno la colonna 'Close'
        :param horizons: lista degli orizzonti temporali in numero di candele
        :param thresholds: dizionario opzionale {h: soglia_percentuale}, default 0.01 per tutti
        """
        import numpy as np

        if thresholds is None:
            thresholds = {h: 0.01 for h in horizons}  # default 1% per ogni orizzonte

        for h in horizons:
            # Rendimento percentuale futuro
            future_return = (df['Close'].shift(-h) - df['Close']) / df['Close']
            df[f'Profit_{h}'] = future_return

            thr = thresholds.get(h, 0.01)  # soglia per questo orizzonte

            # Target coerente: -1, 0, 1
            df[f'Target_{h}'] = 0
            df.loc[future_return > thr, f'Target_{h}'] = 1
            df.loc[future_return < -thr, f'Target_{h}'] = -1

        # Rimuove eventuali NaN finali dovuti a shift
        df.dropna(subset=[f'Profit_{h}' for h in horizons] + [f'Target_{h}' for h in horizons], inplace=True)

        return df

    def show_full_dataframe(self):
        # Stampiamo DataFrame completo
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        #print(self.df_final)
        
        # Profitto percentuale totale (somma dei singoli trade)
        self.df_final['Profit_pct'] = self.df_final['Profit_5'] + self.df_final['Profit_10'] + self.df_final['Profit_20']
        self.df_final['Profit_pct'] = self.df_final['Profit_pct'].fillna(0)
        
        # Calcolo del profitto teorico totale cumulato
        print(f"\nProfitto teorico totale (%) cumulato: {self.df_final['Profit_pct'].sum()*100:.2f}%")
        #self.df_final.to_csv(f"data/{self.symbol}_processed.csv", sep=';', encoding='utf-8')
