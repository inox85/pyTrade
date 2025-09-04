import pandas as pd
import talib
import numpy as np
from tqdm import tqdm
import itertools

class DataPreprocessor:
    def __init__(self, df):
        self.df_origin = df
        self.df_final = df.copy()

    def optimize_macd(self, 
                            fast_range=(5, 20), 
                            slow_range=(26, 100), 
                            signal_range=(5, 20), 
                            horizons=[1, 3, 5],
                            weights=None):
        """
        Ottimizza i parametri del MACD con una grid search su più orizzonti temporali.

        Parametri:
        ----------
        horizons : list[int]
            Lista di orizzonti futuri (in numero di candele).
        weights : list[float] or None
            Pesi per ogni orizzonte (somma = 1). Se None -> pesi uniformi.

        Ritorna:
        --------
        dict con parametri migliori e score.
        """
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

                total_ret = strat_returns.sum()
                volatility = strat_returns.std() if strat_returns.std() != 0 else 1
                sharpe_like = total_ret / volatility

                horizon_scores.append(w * sharpe_like)

            score = sum(horizon_scores)

            if score > best_score:
                best_score = score
                best_params = (fast, slow, signal)

            pbar.update(1)

        pbar.close()



        params =  { 
            "MACD_Fast_length": best_params[0],
            "MACD_Slow_length": best_params[1],
            "MACD_Signal_length": best_params[2],
            "MACD_Score_length": float(best_score)
        }

        print("Migliori parametri MACD:", params)

        return params

    def optimize_mfi(self, horizons=[1, 3, 5]):
        best_score = -np.inf
        best_params = None

        # Range del parametro periodo MFI
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

            # --- Score su più orizzonti temporali ---
            horizon_scores = []
            for h in horizons:
                future_returns = close.pct_change(h).shift(-h) * pos
                score_h = np.nanmean(future_returns)  # media rendimento futuro
                horizon_scores.append(score_h)

            # Score finale: media su tutti gli orizzonti
            score = np.mean(horizon_scores)

            if score > best_score:
                best_score = score
                best_params = tp

            pbar.update(1)

        pbar.close()

        params = {
            "MFI_TimePeriod": int(best_params),
            "MFI_Best_Score": round(float(best_score), 6)
        }

        print("Migliori parametri MFI:", params)
        return params


    def optimize_bbands(self, horizons=[1, 3, 5]):
        best_score = -np.inf
        best_params = None

        # Range dei parametri da ottimizzare
        timeperiod_range = np.arange(5, 30, 1)     # lunghezza media mobile
        nbdevup_range = np.arange(1, 4, 0.1)       # deviazione standard superiore
        nbdevdn_range = np.arange(1, 4, 0.1)       # deviazione standard inferiore
        matype_range = [0, 1, 2, 3, 4, 5, 6, 7, 8]                   # tipo di media mobile (limitato per performance)

        total = len(timeperiod_range) * len(nbdevup_range) * len(nbdevdn_range) * len(matype_range)
        pbar = tqdm(total=total, desc="Ottimizzazione BBANDS")

        close = self.df_final["Close"]

        for tp, up, dn, matype in itertools.product(timeperiod_range, nbdevup_range, nbdevdn_range, matype_range):
            # Calcolo BBANDS
            upper, middle, lower = talib.BBANDS(close, timeperiod=tp, nbdevup=up, nbdevdn=dn, matype=matype)

            # Strategia: long se prezzo < lower, short se prezzo > upper
            pos = np.where(close < lower, 1, np.where(close > upper, -1, 0))

            # --- Score su più orizzonti temporali ---
            horizon_scores = []
            for h in horizons:
                # ritorni futuri a h passi
                future_returns = close.pct_change(h).shift(-h) * pos
                # media o somma (Sharpe o total return)
                score_h = np.nanmean(future_returns)
                horizon_scores.append(score_h)

            # score finale: media pesata dei diversi orizzonti
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
            "BBANDS_Best_Score": round(float(best_score), 6)
        }

        print("Migliori parametri BBANDS:", params)
        return params
    
    def optimize_volume_adx(self, horizons=[1,3,5],
                        vol_ma_range=range(10,51,5),
                        adx_period_range=range(10,21,2),
                        adx_threshold_range=range(10,41,2),
                        vol_mult_range=(1.5,3.0,0.1),
                        metric="cumulative"):
        """
        Ottimizza Volume Anomaly + ADX su più orizzonti temporali includendo tutti i parametri.
        """
        import numpy as np
        import talib
        import itertools
        from tqdm import tqdm

        best_score = -np.inf
        best_params = None

        close = self.df_final["Close"]
        open_ = self.df_final["Open"]
        high = self.df_final["High"]
        low = self.df_final["Low"]
        volume = self.df_final["Volume"]

        vol_multipliers = np.arange(vol_mult_range[0], vol_mult_range[1]+vol_mult_range[2], vol_mult_range[2])

        total = len(vol_ma_range) * len(adx_period_range) * len(adx_threshold_range) * len(vol_multipliers)
        pbar = tqdm(total=total, desc="Ottimizzazione VolAnomaly+ADX")

        for vol_ma, adx_period, adx_thr, vm in itertools.product(vol_ma_range,
                                                                adx_period_range,
                                                                adx_threshold_range,
                                                                vol_multipliers):
            # --- Calcolo indicatori ---
            Vol_MA = talib.SMA(volume, timeperiod=vol_ma)
            ADX_val = talib.ADX(high, low, close, timeperiod=adx_period)

            # --- Segnali ---
            volume_spike = volume > (Vol_MA * vm)
            buy_signal = volume_spike & (close > open_) & (ADX_val > adx_thr)
            sell_signal = volume_spike & (close < open_) & (ADX_val > adx_thr)

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
                best_params = (vol_ma, adx_period, adx_thr, vm)

            pbar.update(1)

        pbar.close()

        result = {
            "Best_Vol_MA": best_params[0],
            "Best_ADX_Timeperiod": best_params[1],
            "Best_ADX_Threshold": best_params[2],
            "Best_Volume_Multiplier": round(float(best_params[3], 2)),
            "Best_Score": round(float(best_score), 6),
            "Metric": metric,
            "Horizons": horizons
        }

        print(result)
        return result



    def generate_dataset(self):

        volume_params = self.optimize_volume_adx()
        mfi_params = self.optimize_mfi()
        bb_params = self.optimize_bbands()
        macd_params = self.optimize_macd()

        macd, macdsignal, macdhist = talib.MACD(self.df_final["Close"], fastperiod=macd_params["MACD_Fast_length"], slowperiod=macd_params["MACD_Slow_length"], signalperiod=macd_params["MACD_Signal_length"])

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
        upper, middle, lower = talib.BBANDS(self.df_final["Close"], timeperiod=bb_params["BBANDS_TimePeriod"], nbdevup=bb_params["BBANDS_NBDevUp"], nbdevdn=bb_params["BBANDS_NBDevDn"], matype=bb_params["BBANDS_MAType"])

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
