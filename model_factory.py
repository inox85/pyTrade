import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

class ModelFactory:
    def __init__(self, df, train_col=["RSI", "MACD", "SMA_20", "SMA_50"]):
        self.df_train = df
        # ====== Features ======
        X = self.df_train[["RSI", "MACD", "SMA_20", "SMA_50"]]

        # ====== Target ======
        y_class = df["signal"]  # -1,0,1
        y_reg = df[["profit_1d", "profit_5d", "profit_10d"]]
