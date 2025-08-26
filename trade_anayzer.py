import pandas as pd
import numpy as np
np.NaN = np.nan  # alias per compatibilità
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PLOT=False

BACK_ROLLING_N = 16

BARS_UPFRONT_FOR_TARGET = 6

class TradeAnalyzer:
    def __init__(self):
        print("TradeAnalyzer initialized")
        self.df_main = None

    def load_data(self, instrument, fetch=False):
        selected_instrument = instrument
        if fetch:
            from datetime import datetime, timedelta
            import dukascopy_python
            from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_GBP_USD, INSTRUMENT_US_PLTR_US_USD

            start = datetime(2010, 1, 1)
            end = datetime(2024, 1, 1)

            interval = dukascopy_python.INTERVAL_DAY_1
            offer_side = dukascopy_python.OFFER_SIDE_BID

            df = dukascopy_python.fetch(
                selected_instrument,
                interval,
                offer_side,
                start,
                end,
            )

            df = df.reset_index()

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
            df.to_csv(f"data/{selected_instrument.split('.')[0]}.csv", index=False)

        self.df_main = pd.read_csv(f"data/{selected_instrument.split('.')[0]}.csv")

        indexZeros = self.df_main[ self.df_main['Volume'] == 0 ].index

        self.df_main.drop(indexZeros , inplace=True)
        self.df_main.loc[(self.df_main["Volume"] == 0 )]
        self.df_main.isna().sum()

    
    def create_tecnical_indicators(self):
        import pandas_ta as ta
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        from scipy.stats import linregress

        self.df_main['ATR'] = self.df_main.ta.atr(length=20)
        self.df_main['RSI'] = self.df_main.ta.rsi()
        self.df_main['Average'] = self.df_main.ta.midprice(length=1) #midprice
        self.df_main['MA40'] = self.df_main.ta.sma(length=40)
        self.df_main['MA80'] = self.df_main.ta.sma(length=80)
        self.df_main['MA160'] = self.df_main.ta.sma(length=160)
        from scipy.stats import linregress
        def get_slope(array):
            y = np.array(array)
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = linregress(x,y)
            return slope
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        backrollingN = BACK_ROLLING_N
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.df_main['slopeMA40'] = self.df_main['MA40'].rolling(window=backrollingN).apply(get_slope, raw=True)
        self.df_main['slopeMA80'] = self.df_main['MA80'].rolling(window=backrollingN).apply(get_slope, raw=True)
        self.df_main['slopeMA160'] = self.df_main['MA160'].rolling(window=backrollingN).apply(get_slope, raw=True)
        self.df_main['AverageSlope'] = self.df_main['Average'].rolling(window=backrollingN).apply(get_slope, raw=True)
        self.df_main['RSISlope'] = self.df_main['RSI'].rolling(window=backrollingN).apply(get_slope, raw=True)

        print(self.df_main.tail(50))

    def create_normalized_tecnical_indicators(self):
        import pandas_ta as ta
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler
        from scipy.stats import linregress

        # Calcolo indicatori
        self.df_main['ATR'] = self.df_main.ta.atr(length=20)
        self.df_main['RSI'] = self.df_main.ta.rsi()
        self.df_main['Average'] = self.df_main.ta.midprice(length=1)  # midprice
        self.df_main['MA40'] = self.df_main.ta.sma(length=40)
        self.df_main['MA80'] = self.df_main.ta.sma(length=80)
        self.df_main['MA160'] = self.df_main.ta.sma(length=160)

        # Funzione per calcolare slope
        def get_slope(array):
            y = np.array(array)
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            return slope

        backrollingN = BACK_ROLLING_N

        self.df_main['slopeMA40'] = self.df_main['MA40'].rolling(window=backrollingN).apply(get_slope, raw=True)
        self.df_main['slopeMA80'] = self.df_main['MA80'].rolling(window=backrollingN).apply(get_slope, raw=True)
        self.df_main['slopeMA160'] = self.df_main['MA160'].rolling(window=backrollingN).apply(get_slope, raw=True)
        self.df_main['AverageSlope'] = self.df_main['Average'].rolling(window=backrollingN).apply(get_slope, raw=True)
        self.df_main['RSISlope'] = self.df_main['RSI'].rolling(window=backrollingN).apply(get_slope, raw=True)

        # -----------------------------
        # Normalizzazione delle colonne necessarie
        # -----------------------------
        scaler = MinMaxScaler()

        columns_to_normalize = [
            'ATR', 'Average', 'MA40', 'MA80', 'MA160',
            'slopeMA40', 'slopeMA80', 'slopeMA160', 'AverageSlope', 'RSISlope'
        ]

        # Applica MinMaxScaler
        self.df_main[columns_to_normalize] = scaler.fit_transform(self.df_main[columns_to_normalize])

        # Stampa di controllo
        print(self.df_main.tail(50))


    def get_target(self):
        pipdiff = 500*1e-5 #for TP
        SLTPRatio = 2 #pipdiff/Ratio gives SL

        def mytarget(barsupfront, df1):
            length = len(df1)
            high = list(df1['High'])
            low = list(df1['Low'])
            close = list(df1['Close'])
            open = list(df1['Open'])
            trendcat = [None] * length

            for line in range (0,length-barsupfront-2):
                valueOpenLow = 0
                valueOpenHigh = 0
                for i in range(1,barsupfront+2):
                    value1 = open[line+1]-low[line+i]
                    value2 = open[line+1]-high[line+i]
                    valueOpenLow = max(value1, valueOpenLow)
                    valueOpenHigh = min(value2, valueOpenHigh)

                    if ( (valueOpenLow >= pipdiff) and (-valueOpenHigh <= (pipdiff/SLTPRatio)) ):
                        trendcat[line] = 1 #-1 downtrend
                        break
                    elif ( (valueOpenLow <= (pipdiff/SLTPRatio)) and (-valueOpenHigh >= pipdiff) ):
                        trendcat[line] = 2 # uptrend
                        break
                    else:
                        trendcat[line] = 0 # no clear trend

            return trendcat
        
        self.df_main['mytarget'] = mytarget(BARS_UPFRONT_FOR_TARGET, self.df_main)
        print(self.df_main.head())
        
        if PLOT:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize = (15,20))
            ax = fig.gca()
            df_model= self.df_main[['Volume', 'ATR', 'RSI', 'Average', 'MA40', 'MA80', 'MA160', 'slopeMA40', 'slopeMA80', 'slopeMA160', 'AverageSlope', 'RSISlope', 'mytarget']]
            df_model.hist(ax = ax)
            plt.show()
    
    def plot_candlestick_with_signals(self, df, start_index, num_rows):
        import mplfinance as mpf

        # Assumi che il DataFrame contenga 'Gmt time', OHLC, 'Volume' e 'predictedTarget'
        df['Gmt time'] = pd.to_datetime(df['Gmt time'], format="%d.%m.%Y %H:%M:%S.%f")
        df.set_index('Gmt time', inplace=True)

        # Colonne per il grafico a candele
        df_candles = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # Creiamo due serie per marker: uno per buy, uno per sell
        buy_signal = np.where(df['predictedTarget'] == 2, df['Low']*0.995, np.nan)   # leggermente sotto il minimo della candela
        sell_signal = np.where(df['predictedTarget'] == 1, df['High']*1.005, np.nan) # leggermente sopra il massimo della candela

        # Aggiungi i marker al grafico1
        apds = [
            mpf.make_addplot(buy_signal, type='scatter', markersize=20, marker='^', color='green'),
            mpf.make_addplot(sell_signal, type='scatter', markersize=20, marker='v', color='red')
        ]

        # Plot
        mpf.plot(
            df_candles,
            type='candle',
            volume=True,
            style='yahoo',
            title='Grafico a Candele con segnali',
            ylabel='Prezzo',
            ylabel_lower='Volume',
            addplot=apds
        )

     
    def train_model(self):
        df_model=self.df_main.dropna()

        attributes=['ATR', 'RSI', 'Average', 'MA40', 'MA80', 'MA160', 'slopeMA40', 'slopeMA80', 'slopeMA160', 'AverageSlope', 'RSISlope']
        X = df_model[attributes]
        y = df_model["mytarget"]

        print(X)

        train_set_ratio = 0.5
        #sequential sampling
        train_index = int(train_set_ratio * len(X))
        X_train, X_test = X[:train_index], X[train_index:]
        y_train, y_test = y[:train_index], y[train_index:]

        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score
        model = XGBClassifier()
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        acc_train = accuracy_score(y_train, pred_train)
        acc_test = accuracy_score(y_test, pred_test)

        # Copia indipendente del DataFrame
        df_model = self.df_main.dropna().copy()

        # Indici del test set
        train_index = int(train_set_ratio * len(df_model))
        test_df = df_model.iloc[train_index:].copy()  # il test set come DataFrame separato

        
        # Se la lunghezza del test set coincide con le predizioni
        if len(test_df) == len(pred_test):
            df_model.loc[test_df.index, 'predictedTarget'] = pred_test
        else:
            raise ValueError(f"Lunghezza test set ({len(test_df)}) != lunghezza predizioni ({len(pred_test)})")

        # Verifica
        print(df_model[['mytarget','predictedTarget']].tail(len(test_df)))

        for row in df_model.itertuples():
            if row.predictedTarget == 2:
                print(row.Index, row)   # Index = indice della riga

        print('****Train Results****')
        print("Accuracy: {:.4%}".format(acc_train))
        print('****Test Results****')
        print("Accuracy: {:.4%}".format(acc_test))
        
        self.model = model
        self.attributes = attributes

        #self.plot_candlestick_with_signals(df_model, start_index=train_index, num_rows=10)

        