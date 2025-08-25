import pandas as pd
import numpy as np
np.NaN = np.nan  # alias per compatibilità


class TradeAnalyzer:
    def __init__(self):
        print("TradeAnalyzer initialized")
        self.df_main = None

    def download_data(self, instrument):
        from datetime import datetime, timedelta
        import dukascopy_python
        from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_GBP_USD, INSTRUMENT_US_PLTR_US_USD

        start = datetime(2017, 1, 1)
        end = datetime(2024, 1, 1)
        selected_instrument = instrument
        interval = dukascopy_python.INTERVAL_DAY_1
        offer_side = dukascopy_python.OFFER_SIDE_BID

        df = dukascopy_python.fetch(
            selected_instrument,
            interval,
            offer_side,
            start,
            end,
        )

        df.reset_index(inplace=True)

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

                #Check if any zero volumes are available
        indexZeros = self.df_main[ self.df_main['Volume'] == 0 ].index

        self.df_main.drop(indexZeros , inplace=True)
        self.df_main.loc[(self.df_main["Volume"] == 0 )]
        self.df_main.isna().sum()
        
        #print(self.df_main)

        #print(self.df_main.isna().sum())
    
    def create_tecnical_indicators(self):
        import pandas_ta as ta
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
        backrollingN = 6
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.df_main['slopeMA40'] = self.df_main['MA40'].rolling(window=backrollingN).apply(get_slope, raw=True)
        self.df_main['slopeMA80'] = self.df_main['MA80'].rolling(window=backrollingN).apply(get_slope, raw=True)
        self.df_main['slopeMA160'] = self.df_main['MA160'].rolling(window=backrollingN).apply(get_slope, raw=True)
        self.df_main['AverageSlope'] = self.df_main['Average'].rolling(window=backrollingN).apply(get_slope, raw=True)
        self.df_main['RSISlope'] = self.df_main['RSI'].rolling(window=backrollingN).apply(get_slope, raw=True)

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
        
        self.df_main['mytarget'] = mytarget(16, self.df_main)
        print(self.df_main.head())

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize = (15,20))
        ax = fig.gca()
        df_model= self.df_main[['Volume', 'ATR', 'RSI', 'Average', 'MA40', 'MA80', 'MA160', 'slopeMA40', 'slopeMA80', 'slopeMA160', 'AverageSlope', 'RSISlope', 'mytarget']]
        df_model.hist(ax = ax)
        plt.show()