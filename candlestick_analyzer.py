import numpy as np
np.NaN = np.nan  # alias per compatibilità
import pandas as pd
import pandas_ta as ta
from tqdm import tqdm
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backtesting import Strategy
from backtesting import Backtest

tqdm.pandas()
class CandlestickAnalyzer:
    def __init__(self):
        self.dataframes = []
        np.NaN = np.nan  # alias per compatibilità
    
    def read_csv_to_dataframe(self, file_path):
        df = pd.read_csv(file_path)
        df["Gmt time"] = df["Gmt time"].str.replace(".000", "")
        df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S')
        df = df[df.High != df.Low]
        df.set_index("Gmt time", inplace=True)
        return df

    def read_data_folder(self, folder_path="./data"):
        file_names = []
        for file_name in tqdm(os.listdir(folder_path)):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = self.read_csv_to_dataframe(file_path)
                self.dataframes.append(df)
                file_names.append(file_name)
        return self.dataframes, file_names

    def total_signal(self, df, current_candle):
        current_pos = df.index.get_loc(current_candle)
        c0 = df['Open'].iloc[current_pos] > df['Close'].iloc[current_pos]
        # Condition 1: The high is greater than the high of the previous day
        c1 = df['High'].iloc[current_pos] > df['High'].iloc[current_pos - 1]
        # Condition 2: The low is less than the low of the previous day
        c2 = df['Low'].iloc[current_pos] < df['Low'].iloc[current_pos - 1]
        # Condition 3: The close of the Outside Bar is less than the low of the previous day
        c3 = df['Close'].iloc[current_pos] < df['Low'].iloc[current_pos - 1]

        if c0 and c1 and c2 and c3:
            print("Long signal found at index:", current_pos, "Date:", df.index[current_pos])
            return 2  # Signal for entering a Long trade at the open of the next bar
        
        c0 = df['Open'].iloc[current_pos] < df['Close'].iloc[current_pos]
        # Condition 1: The high is greater than the high of the previous day
        c1 = df['Low'].iloc[current_pos] < df['Low'].iloc[current_pos - 1]
        # Condition 2: The low is less than the low of the previous day
        c2 = df['High'].iloc[current_pos] > df['High'].iloc[current_pos - 1]
        # Condition 3: The close of the Outside Bar is less than the low of the previous day
        c3 = df['Close'].iloc[current_pos] > df['High'].iloc[current_pos - 1]
        
        if c0 and c1 and c2 and c3:
            print("Short signal found at index:", current_pos, "Date:", df.index[current_pos])
            return 1
        
        return 0

    def add_total_signal(self,df):
        df['TotalSignal'] = df.progress_apply(lambda row: self.total_signal(df, row.name), axis=1)#.shift(1)
        return df

    def add_pointpos_column(self, df, signal_column):
        """
        Adds a 'pointpos' column to the DataFrame to indicate the position of support and resistance points.
        
        Parameters:
        df (DataFrame): DataFrame containing the stock data with the specified SR column, 'Low', and 'High' columns.
        sr_column (str): The name of the column to consider for the SR (support/resistance) points.
        
        Returns:
        DataFrame: The original DataFrame with an additional 'pointpos' column.
        """
        def pointpos(row):
            if row[signal_column] == 2:
                return row['Low'] - 1e-4
            elif row[signal_column] == 1:
                return row['High'] + 1e-4
            else:
                return np.nan

        df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)
        return df

    def plot_candlestick_with_signals(self, df, start_index, num_rows):
        """
        Plots a candlestick chart with signal points.
        
        Parameters:
        df (DataFrame): DataFrame containing the stock data with 'Open', 'High', 'Low', 'Close', and 'pointpos' columns.
        start_index (int): The starting index for the subset of data to plot.
        num_rows (int): The number of rows of data to plot.
        
        Returns:
        None
        """

        import plotly.express as px
        import plotly.io as pio

        pio.renderers.default = "browser"

        df_subset = df[start_index:start_index + num_rows]
        
        fig = make_subplots(rows=1, cols=1)
        
        fig.add_trace(go.Candlestick(x=df_subset.index,
                                    open=df_subset['Open'],
                                    high=df_subset['High'],
                                    low=df_subset['Low'],
                                    close=df_subset['Close'],
                                    name='Candlesticks'),
                    row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df_subset.index, y=df_subset['pointpos'], mode="markers",
                                marker=dict(size=10, color="MediumPurple", symbol='circle'),
                                name="Entry Points"),
                    row=1, col=1)
        
        fig.update_layout(
            width=1800, 
            height=900, 
            plot_bgcolor='black',
            paper_bgcolor='black',
            font=dict(color='white'),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            showlegend=True,
            legend=dict(
                x=0.01,
                y=0.99,
                traceorder="normal",
                font=dict(
                    family="sans-serif",
                    size=12,
                    color="white"
                ),
                bgcolor="black",
                bordercolor="gray",
                borderwidth=2
            )
        )
        
        #fig.show()
        fig.write_html("grafico.html", auto_open=True)

    def execute(self):

        dataframes, file_names = self.read_data_folder()

        for i, df in enumerate(dataframes):
            print("working on dataframe ", i, "...")
            df = self.add_total_signal(df)
            df = self.add_pointpos_column(df, "TotalSignal")
            dataframes[i] = df  # Update the dataframe in the list
            sum([frame["TotalSignal"].value_counts() for frame in dataframes], start=0)
            self.plot_candlestick_with_signals(dataframes[0], start_index=0, num_rows=len(dataframes[0]))

        return dataframes
    
from backtesting import Strategy, Backtest


class Strategy_01(Strategy):
    mysize = 0.1
    slperc = 0.02
    tpperc = 0.04
    signal_column = 'TotalSignal'  # default, si può cambiare

    def init(self):
        # prendi la colonna segnali dal dataframe che Backtesting.py ti passa
        self.signal = self.I(lambda: self.data.df[self.signal_column].values)

    def next(self):
        current_signal = self.signal[-1]
        current_close = self.data.Close[-1]

        if current_signal == 2 and not self.position:
            sl = current_close * (1 - self.slperc)
            tp = current_close * (1 + self.tpperc)
            self.buy(size=self.mysize, sl=sl, tp=tp)

        elif current_signal == 1 and not self.position:
            sl = current_close * (1 + self.slperc)
            tp = current_close * (1 - self.tpperc)
            self.sell(size=self.mysize, sl=sl, tp=tp)

class Strategy_02(Strategy):
    mysize = 0.1
    slperc = 0.02
    tpperc = 0.04
    signal_column = 'TotalSignal'  # default, si può cambiare

    def init(self):
        # prendi la colonna segnali dal dataframe che Backtesting.py ti passa
        self.signal = self.I(lambda: self.data.df[self.signal_column].values)

    def calculate_stop_loss(self, entry_price, pips=200, pip_value=0.0001, direction="long"):
        """
        Calculate the stop loss distance given the entry price, number of pips, and pip value.
        
        Parameters:
        entry_price (float): The price at which the trade is entered.
        pips (int): The number of pips for the stop loss. Default is 200.
        pip_value (float): The value of one pip. Default is 0.0001 for most currency pairs.
        direction (str): 'long' or 'short' to indicate the trade direction.
        
        Returns:
        float: The stop loss price.
        """
        sl_distance = pips * pip_value
        if direction == "long":
            stop_loss_price = entry_price - sl_distance
        elif direction == "short":
            stop_loss_price = entry_price + sl_distance
        else:
            raise ValueError("direction must be 'long' or 'short'")

        return stop_loss_price

    def next(self):
        super().next()

        # Check if any trades are winning and close them
        for trade in self.trades:
            if trade.pl > 0:
                trade.close()

        # Handle new signals
        if self.signal[-1] == 2 and not self.position:
            current_close = self.data.Close[-1]
            sl = self.calculate_stop_loss(entry_price=current_close, pips=250, pip_value=0.0001, direction="long")
            self.buy(size=self.mysize, sl=sl)

        elif self.signal[-1] == 1 and not self.position:
            current_close = self.data.Close[-1]
            sl = self.calculate_stop_loss(entry_price=current_close, pips=250, pip_value=0.0001, direction="short")
            self.sell(size=self.mysize, sl=sl)




def run_backtest(template,  df, signal_column='TotalSignal'):
    backtest = Backtest(df, template, cash=10_000, commission=.002)
    stats = backtest.run()
    return stats

def show_results():
    agg_returns = sum([r["Return [%]"] for r in self.results])
    num_trades = sum([r["# Trades"] for r in self.results])
    max_drawdown = min([r["Max. Drawdown [%]"] for r in self.results])
    avg_drawdown = sum([r["Avg. Drawdown [%]"] for r in self.results]) / len(self.results)

    win_rate = sum([r["Win Rate [%]"] for r in self.results]) / len(self.results)
    best_trade = max([r["Best Trade [%]"] for r in self.results])
    worst_trade = min([r["Worst Trade [%]"] for r in self.results])
    avg_trade = sum([r["Avg. Trade [%]"] for r in self.results]) / len(self.results)
    #max_trade_duration = max([r["Max. Trade Duration"] for r in results])
    #avg_trade_duration = sum([r["Avg. Trade Duration"] for r in results]) / len(results)

    print(f"Aggregated Returns: {agg_returns:.2f}%")
    print(f"Number of Trades: {num_trades}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Average Drawdown: {avg_drawdown:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Best Trade: {best_trade:.2f}%")
    print(f"Worst Trade: {worst_trade:.2f}%")
    print(f"Average Trade: {avg_trade:.2f}%")