import pandas as pd
import talib
import numpy as np
from tqdm import tqdm
import itertools
import json
import os

class DataPreprocessor:
    def __init__(self, df, symbol, interval):      
        self.interval = interval
        self.df_origin = df
        self.df_final = df.copy()
        self.symbol = symbol
        self.params_file = "params/params.json"
        data = self.load_json_safe(self.params_file, True)

        if self.symbol not in data:
            print("Parametri non trovati per il simbolo:", self.symbol)
            print("Il campo verrà creato")
            data[self.symbol] = {}
            data[self.symbol][self.interval] = {}
            with open(self.params_file, "w") as f:
                json.dump(data, f, indent=4)
        
        self.config_params = data[self.symbol][self.interval]
        print(self.config_params)
    

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
            data[self.symbol][field] = params
        
        with open(self.params_file, "w") as f:
            json.dump(data, f, indent=4)

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

        grid = itertools.product(
            range(fast_range[0], fast_range[1]),
            range(slow_range[0], slow_range[1]),
            range(signal_range[0], signal_range[1])
        )

        total = (fast_range[1]-fast_range[0]) * (slow_range[1]-slow_range[0]) * (signal_range[1]-signal_range[0])
        pbar = tqdm(total=total, desc="Ottimizzazione MACD multi-horizon")

        for fast, slow, signal in grid:
            if fast >= slow:
                pbar.update(1)
                continue

            macd, signal_line, hist = talib.MACD(
                self.df_final["Close"],
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal
            )

            pos = (hist > 0).astype(int)

            # valutazione su più orizzonti
            horizon_scores = []
            for h, w in zip(horizons, weights):
                future_returns = self.df_final["Close"].pct_change(h).shift(-h)
                strat_returns = future_returns * pos

                if metric == "cumulative":
                    score_h = strat_returns.sum() * 100
                elif metric == "sharpe":
                    total_ret = strat_returns.sum()
                    volatility = strat_returns.std() if strat_returns.std() != 0 else 1
                    score_h = total_ret / volatility
                elif metric == "accuracy":
                    correct_signals = np.sum((strat_returns > 0) & (pos != 0))
                    total_signals = np.sum(pos != 0)
                    score_h = correct_signals / total_signals if total_signals > 0 else 0
                else:
                    raise ValueError("Metric deve essere 'cumulative', 'sharpe' o 'accuracy'")

                horizon_scores.append(w * score_h)

            score = sum(horizon_scores)

            if score > best_score:
                best_score = score
                best_params = (fast, slow, signal)

            pbar.update(1)

        pbar.close()

        params = { 
            "MACD_Fast_length": best_params[0],
            "MACD_Slow_length": best_params[1],
            "MACD_Signal_length": best_params[2],
            "Best_Score": round(float(best_score), 6),
            "Metric": metric,
            "Horizons": horizons
        }

        print("Migliori parametri MACD:", params)

        self.save_params("macd", params)

        return params


    def optimize_mfi(self, horizons=[1, 3, 5], metric="cumulative"):
        """
        Ottimizza i parametri del MFI (Money Flow Index).

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
        from tqdm import tqdm

        best_score = -np.inf
        best_params = None

        timeperiod_range = np.arange(5, 50, 1)

        total = len(timeperiod_range)
        pbar = tqdm(total=total, desc="Ottimizzazione MFI")

        close = self.df_final["Close"]
        high = self.df_final["High"]
        low = self.df_final["Low"]
        volume = self.df_final["Volume"]

        for tp in timeperiod_range:
            # Calcolo MFI
            mfi = talib.MFI(high, low, close, volume, timeperiod=tp)

            # Strategia: long se MFI < 20, short se MFI > 80
            pos = np.where(mfi < 20, 1, np.where(mfi > 80, -1, 0))

            horizon_scores = []
            for h in horizons:
                future_returns = close.pct_change(h).shift(-h) * pos

                if metric == "cumulative":
                    score_h = np.nansum(future_returns) * 100
                elif metric == "sharpe":
                    mean_ret = np.nanmean(future_returns)
                    std_ret = np.nanstd(future_returns)
                    score_h = mean_ret / std_ret if std_ret != 0 else -np.inf
                elif metric == "accuracy":
                    correct_signals = np.sum((future_returns > 0) & (pos != 0))
                    total_signals = np.sum(pos != 0)
                    score_h = correct_signals / total_signals if total_signals > 0 else 0
                else:
                    raise ValueError("Metric deve essere 'cumulative', 'sharpe' o 'accuracy'")

                horizon_scores.append(score_h)

            score = np.mean(horizon_scores)

            if score > best_score:
                best_score = score
                best_params = tp

            pbar.update(1)

        pbar.close()

        params = {
            "MFI_TimePeriod": int(best_params),
            "Best_Score": round(float(best_score), 6),
            "Metric": metric,
            "Horizons": horizons
        }

        self.save_params("mfi", params)
        print("Migliori parametri MFI:", params)
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
        nbdevup_range = np.arange(1, 4, 0.1)
        nbdevdn_range = np.arange(1, 4, 0.1)
        matype_range = [0, 1, 2, 3, 4]  # limito i tipi di MA per performance

        total = len(timeperiod_range) * len(nbdevup_range) * len(nbdevdn_range) * len(matype_range)
        pbar = tqdm(total=total, desc="Ottimizzazione BBANDS")

        close = self.df_final["Close"]

        for tp, up, dn, matype in itertools.product(timeperiod_range, nbdevup_range, nbdevdn_range, matype_range):
            upper, middle, lower = talib.BBANDS(close, timeperiod=tp, nbdevup=up, nbdevdn=dn, matype=matype)

            # Strategia: long se prezzo < lower, short se prezzo > upper
            pos = np.where(close < lower, 1, np.where(close > upper, -1, 0))

            horizon_scores = []
            for h in horizons:
                future_returns = close.pct_change(h).shift(-h) * pos

                if metric == "cumulative":
                    score_h = np.nansum(future_returns) * 100
                elif metric == "sharpe":
                    mean_ret = np.nanmean(future_returns)
                    std_ret = np.nanstd(future_returns)
                    score_h = mean_ret / std_ret if std_ret != 0 else -np.inf
                elif metric == "accuracy":
                    correct_signals = np.sum((future_returns > 0) & (pos != 0))
                    total_signals = np.sum(pos != 0)
                    score_h = correct_signals / total_signals if total_signals > 0 else 0
                else:
                    raise ValueError("Metric deve essere 'cumulative', 'sharpe' o 'accuracy'")

                horizon_scores.append(score_h)

            score = np.mean(horizon_scores)

            if score > best_score:
                best_score = score
                best_params = (tp, up, dn, matype)

            pbar.update(1)

        pbar.close()

        params = {
            "BBANDS_TimePeriod": int(best_params[0]),
            "BBANDS_NBDevUp": round(float(best_params[1]), 2),
            "BBANDS_NBDevDn": round(float(best_params[2]), 2),
            "BBANDS_MAType": int(best_params[3]),
            "Best_Score": round(float(best_score), 6),
            "Metric": metric,
            "Horizons": horizons
        }

        self.save_params("bBands", params)
        print("Migliori parametri BBANDS:", params)
        return params


# --------------------------
# Esempio: dati OHLCV
# df deve avere colonne: ['Open', 'High', 'Low', 'Close', 'Volume']

# Funzione di ottimizzazione
# --------------------------
    def optimize_volume_adx(self, 
                        horizons=[1,3,5],
                        donchian_range=np.arange(5,16,1),
                        vol_ma_range=np.arange(5,16,0.1),
                        adx_period_range=np.arange(5,15,1),
                        adx_threshold_range=np.arange(5,16,0.1),
                        vol_mult_range=np.arange(1.05,1.5,0.1),
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
        pbar = tqdm(total=total, desc="Ottimizzazione Volume+ADX", ncols=150)

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
            num_signals = np.sum(pos != 0)
            if num_signals == 0:
                pbar.update(1)
                continue

            # --- Score multi-orizzonte ---
            horizon_scores = []
            for h in horizons:
                future_returns = np.log(close.shift(-h) / close) * pos
                if metric == "cumulative":
                    score_h = np.nansum(future_returns) / num_signals * 100
                elif metric == "sharpe":
                    mean_ret = np.nanmean(future_returns)
                    std_ret = np.nanstd(future_returns)
                    score_h = mean_ret / std_ret if std_ret != 0 else -np.inf
                elif metric == "accuracy":
                    correct_signals = np.sum((future_returns > 0) & (pos != 0))
                    score_h = correct_signals / num_signals
                else:
                    raise ValueError("Metric deve essere 'cumulative', 'sharpe' o 'accuracy'")
                horizon_scores.append(score_h)

            score = np.mean(horizon_scores)

            if score > best_score:
                best_score = score
                best_params = (donchian_w, vol_ma, adx_period, adx_thr, vm)

            # Aggiornamento progress bar
            postfix_str = (f"best_score={best_score:.5f} signals={num_signals}")
            pbar.set_postfix_str(postfix_str)
            pbar.update(1)

        pbar.close()

        if best_params is None:
            raise RuntimeError("Nessun segnale valido trovato!")

        params = {
            "Best_Donchian_Window": int(best_params[0]),
            "Best_Vol_MA": int(best_params[1]),
            "Best_ADX_Timeperiod": int(best_params[2]),
            "Best_ADX_Threshold": round(float(best_params[3]),2),
            "Best_Volume_Multiplier": round(float(best_params[4]),2),
            "Best_Score": float(best_score),
            "Metric": metric,
            "Horizons": horizons
        }

        self.save_params("volumeAdx", params)
        print("Migliori parametri Volume+ADX:", params)
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
        pbar = tqdm(total=total, desc="Ottimizzazione RSI")

        for period, low_thr, high_thr in itertools.product(rsi_period_range,
                                                        rsi_low_range,
                                                        rsi_high_range):
            if low_thr >= high_thr:  # soglie non valide
                continue

            # --- RSI ---
            rsi = talib.RSI(close, timeperiod=period)

            # --- Segnali ---
            buy_signal = rsi < low_thr
            sell_signal = rsi > high_thr
            pos = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))

            # --- Score multi-orizzonte ---
            horizon_scores = []
            for h in horizons:
                future_returns = close.pct_change(h).shift(-h) * pos
                if metric == "cumulative":
                    score_h = np.nansum(future_returns) * 100
                elif metric == "sharpe":
                    mean_return = np.nanmean(future_returns)
                    std_return = np.nanstd(future_returns)
                    score_h = mean_return / std_return if std_return != 0 else -np.inf
                elif metric == "accuracy":
                    correct_signals = np.sum((future_returns > 0) & (pos != 0))
                    total_signals = np.sum(pos != 0)
                    score_h = correct_signals / total_signals if total_signals > 0 else 0
                else:
                    raise ValueError("Metric deve essere 'cumulative', 'sharpe' o 'accuracy'")
                horizon_scores.append(score_h)

            score = np.mean(horizon_scores)

            if score > best_score:
                best_score = score
                best_params = (period, low_thr, high_thr)

            pbar.update(1)

        pbar.close()

        params = {
            "Best_RSI_Period": best_params[0],
            "Best_RSI_Low": best_params[1],
            "Best_RSI_High": best_params[2],
            "Best_Score": round(float(best_score),6),
            "Metric": metric,
            "Horizons": horizons
        }
        self.save_params("rsi", params)

        return params
    
    def optimize_trade_count_norm(self, horizons=[1,3,5],
                              window_range=range(2, 51),
                              metric="cumulative"):
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
        pbar = tqdm(total=total, desc="Ottimizzazione trade_count_norm")

        for window in window_range:
            # --- Calcolo trade_count_norm ---
            trade_count_norm = trade_count / trade_count.rolling(window).mean()

            # Segnali basati su trade_count_norm > 1 => attività alta, < 1 => bassa
            buy_signal = trade_count_norm > 1
            sell_signal = trade_count_norm < 1
            pos = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))

            # --- Score multi-orizzonte ---
            horizon_scores = []
            for h in horizons:
                future_returns = close.pct_change(h).shift(-h) * pos
                if metric == "cumulative":
                    score_h = np.nansum(future_returns) * 100
                elif metric == "sharpe":
                    mean_return = np.nanmean(future_returns)
                    std_return = np.nanstd(future_returns)
                    score_h = mean_return / std_return if std_return != 0 else -np.inf
                elif metric == "accuracy":
                    correct_signals = np.sum((future_returns > 0) & (pos != 0))
                    total_signals = np.sum(pos != 0)
                    score_h = correct_signals / total_signals if total_signals > 0 else 0
                else:
                    raise ValueError("Metric deve essere 'cumulative', 'sharpe' o 'accuracy'")
                horizon_scores.append(score_h)

            score = np.mean(horizon_scores)

            if score > best_score:
                best_score = score
                best_window = window

            pbar.update(1)

        pbar.close()

        params = {
            "Best_TradeCountNorm_Window": best_window,
            "Best_Score": round(float(best_score), 6),
            "Metric": metric,
            "Horizons": horizons
        }

        self.save_params("trade_count_norm", params)
        return params


    def generate_dataset(self):

        self.volume_params = self.optimize_volume_adx()
        self.rsi_params = self.optimize_rsi()
        self.macd_params = self.optimize_macd()
        self.mfi_params = self.optimize_mfi()
        self.bb_params = self.optimize_bbands()
        

        macd, macdsignal, macdhist = talib.MACD(self.df_final["Close"], fastperiod=self.macd_params["MACD_Fast_length"], slowperiod=self.macd_params["MACD_Slow_length"], signalperiod=self.macd_params["MACD_Signal_length"])

        #macd, macdsignal, macdhist = talib.MACD(self.df_final["Close"], fastperiod=5, slowperiod=14, signalperiod=28)

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

        # --- Bollinger Bands ---
        upper, middle, lower = talib.BBANDS(self.df_final["Close"], timeperiod=self.bb_params["BBANDS_TimePeriod"], nbdevup=self.bb_params["BBANDS_NBDevUp"], nbdevdn=self.bb_params["BBANDS_NBDevDn"], matype=self.bb_params["BBANDS_MAType"])

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


        # result = {
        #     "Best_Donchian_Window": int(best_params[0]),
        #     "Best_Vol_MA": int(best_params[1]),
        #     "Best_ADX_Timeperiod": int(best_params[2]),
        #     "Best_ADX_Threshold": round(float(best_params[3]), 2),
        #     "Best_Volume_Multiplier": round(float(best_params[4]), 2),
        #     "Best_Score": round(float(best_score), 6),
        #     "Metric": metric,
        #     "Horizons": horizons
        # }
        1
        # Volume Anomaly
        sma_volume = talib.SMA(self.df_final['Volume'], timeperiod=self.volume_params["Best_Vol_MA"])

        for vol_coeff in [1.5, 2.0, 3.0]:
            volume_spike = self.df_final['Volume'] > (sma_volume * vol_coeff)
            self.df_final[f'VolAnomaly_Buy_{vol_coeff}'] = volume_spike & (self.df_final['Close'] > self.df_final['Open'])
            self.df_final[f'VolAnomaly_Sell_{vol_coeff}'] = volume_spike & (self.df_final['Close'] < self.df_final['Open'])

        # -------------------
        # Breakout con conferma volumi + ADX
        # -------------------
        self.df_final["Donchian_High"] = self.df_final["High"].rolling(window=self.volume_params["Best_Donchian_Window"]).max()
        self.df_final["Donchian_Low"] = self.df_final["Low"].rolling(window=self.volume_params["Best_Donchian_Window"]).min()
        self.df_final["Vol_MA"] = talib.SMA(self.df_final["Volume"], timeperiod=self.volume_params["Best_Vol_MA"])
        self.df_final["ADX"] = talib.ADX(self.df_final["High"], self.df_final["Low"], self.df_final["Close"], timeperiod=self.volume_params["Best_ADX_Timeperiod"])
        adx_threshold = self.volume_params["Best_ADX_Threshold"]

        self.df_final["Breakout_Up"] = (
            (self.df_final["Close"] > self.df_final["Donchian_High"].shift(1)) &
            (self.df_final["Volume"] > self.df_final["Vol_MA"]) &
            (self.df_final["ADX"] > adx_threshold)
        )

        self.df_final["Breakout_Down"] = (
            (self.df_final["Close"] < self.df_final["Donchian_Low"].shift(1)) &
            (self.df_final["Volume"] > self.df_final["Vol_MA"]) &
            (self.df_final["ADX"] > adx_threshold)
        )

        
        # # === 1. Differenza Close - VWAP
        # df["close_minus_vwap"] = df["close"] - df["vwap"]

        # # === 2. Rapporto Close/VWAP
        # df["close_div_vwap"] = df["close"] / df["vwap"]

        # # === 3. Dimensione media degli scambi (volume medio per trade)
        # df["avg_trade_size"] = df["volume"] / df["trade_count"]

        # # === 4. Normalizzazione del trade_count (rispetto alla media mobile 3)
        # df["trade_count_norm"] = df["trade_count"] / df["trade_count"].rolling(3).mean()


    def generate_targets(self, horizons=[5, 10, 20], threshold=0.01):
        """
        Genera target multi-orizzonte:
        - Target_{h}: Buy (1), Sell (-1), Hold (0) basati su una soglia percentuale
        - Profit_{h}: rendimento percentuale nei prossimi h passi

        :param horizons: lista degli orizzonti temporali in numero di candele
        :param threshold: soglia percentuale per classificare Buy/Sell/Hold
        """
        for h in horizons:
            # Rendimento percentuale futuro
            future_return = (self.df_final['Close'].shift(-h) - self.df_final['Close']) / self.df_final['Close']

            # Profitto percentuale
            self.df_final[f'Profit_{h}'] = future_return

            # Target classificazione
            self.df_final[f'Target_{h}'] = np.where(
                future_return > threshold, 1,     # Buy
                np.where(future_return < -threshold, -1, 0)  # Sell o Hold
            )

        #return self.df_final

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
