import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
from itertools import combinations
from tqdm import tqdm

"""
MACD_diff → importanza 0 su tutti i target, quindi inutile.

Engulfing_direction → quasi sempre <0.02, contribuisce poco al modello.

Vol_Ratio → importanza molto bassa, soprattutto sui target più lunghi.

EMA_5_slope_pct → contribuisce poco, spesso <0.03.

OBV_slope → importanza bassa, tranne qualche target breve.

Engulfing_strength → importanza bassa su orizzonti lunghi, può generare rumore.

Trade_Count_Norm_5 / Avg_Trade_Size → utile solo sui target brevi (Profit_1), meno sugli altri.

Donchian_Close_HighPct / LowPct → contributo molto marginale.

RSI_norm → importanza bassa, soprattutto su Profit_20 e Profit_50.
"""

X_col = [
    # Prezzo/volume base
    'Close', 'Volume',

    # EMA
    'EMA_5', 'EMA_20',
    'EMA_50', 'EMA_100',
    'EMA_5_slope_pct',
    'EMA_20_slope_pct',
    'EMA_5_x_EMA_20_diff',
    'EMA_20_x_EMA_50_diff',

    # MACD
    'MACD_hist',
    'MACD_diff',
    'MACD_slope',

    # Bollinger
    'BB_Width',
    'BB_Percent_b',

    # OBV
    'OBV_slope',
    'OBV_momentum',

    # Donchian
    'Donchian_Close_HighPct',
    'Donchian_Close_LowPct',

    # Pattern
    'Engulfing_strength',
    'Engulfing_direction',

    # Momentum
    'RSI_norm',
    'ADX', 'ADX_Slope',

    # Volumi extra
    'Vol_Ratio',
    'Avg_Trade_Size',
    'Trade_Count_Norm_5',

    # VWAP
    'Close_VWAP_Diff'
]

y_profit_col= ['Profit_1','Profit_5','Profit_10','Profit_20','Profit_50']
y_target_col  = ['Target_1','Target_5','Target_10','Target_20','Target_50']
y_cumulative_col = ['Cumulative_1','Cumulative_5','Cumulative_10','Cumulative_20','Cumulative_50']

class ModelFactory:
    def __init__(self, df, train_col=X_col, profit_col=y_profit_col,
                 target_col=y_target_col, cumulative_col=y_cumulative_col):

        self.df_train = df

        # ====== Features ======
        self.X = self.df_train[train_col]

        # ====== Scaler per le features ======
        self.scaler_X = StandardScaler()
        self.X_scaled = pd.DataFrame(
            self.scaler_X.fit_transform(self.X),
            columns=self.X.columns,
            index=self.X.index
        )

        self.X_scaled.to_csv(f"datasets/X_scaled.csv", sep=';', encoding='utf-8')

        # ====== Target regressione: profit ======
        self.y_profit = self.df_train[profit_col]
        self.scaler_profit = StandardScaler()
        self.y_profit_scaled = pd.DataFrame(
            self.scaler_profit.fit_transform(self.y_profit),
            columns=self.y_profit.columns,
            index=self.y_profit.index
        )

        # ====== Target regressione: cumulative ======
        self.y_cumulative = self.df_train[cumulative_col]
        self.scaler_cumulative = StandardScaler()
        self.y_cumulative_scaled = pd.DataFrame(
            self.scaler_cumulative.fit_transform(self.y_cumulative),
            columns=self.y_cumulative.columns,
            index=self.y_cumulative.index
        )

        # ====== Target classificazione: target ======
        self.y_target = self.df_train[target_col]

        best_results = self.feature_subset_search(max_features=4)  # es. fino a 5 feature
        print(best_results.head(10))
        # ====== Generazione modelli ======
        #self.generate_models()


    def generate_models(self):
        # --- Profit regressione ---
        X_train, X_test, y_train, y_test = train_test_split(self.X_scaled, self.y_profit_scaled, test_size=0.2, shuffle=False)

        profit_model = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=5))
        profit_model.fit(X_train, y_train)

        for i, col in enumerate(self.y_profit_scaled.columns):
            importances = profit_model.estimators_[i].feature_importances_

            # usare X_scaled.columns come index
            feat_imp = pd.Series(importances, index=self.X_scaled.columns).sort_values(ascending=False)

            print(f"Feature importance per {col}:")
            print(feat_imp)

            # plot
            feat_imp.plot(kind='bar', title=f"Feature Importance {col}")
            plt.show()


        pred = profit_model.predict(X_test)
        pred_df = pd.DataFrame(pred, columns=y_test.columns, index=y_test.index)

        print("=== Profit Metrics ===")
        for col in y_test.columns:
            mse = mean_squared_error(y_test[col], pred_df[col])
            mae = mean_absolute_error(y_test[col], pred_df[col])
            r2 = r2_score(y_test[col], pred_df[col])
            print(f"{col}: MSE={mse:.6f}, MAE={mae:.6f}, R2={r2:.4f}")
        mse_total = mean_squared_error(y_test, pred_df)
        mae_total = mean_absolute_error(y_test, pred_df)
        print(f"MSE totale: {mse_total:.6f}, MAE totale: {mae_total:.6f}")

        direction_accuracy = np.mean(np.sign(y_test.values) == np.sign(pred_df.values))
        print(f"Accuratezza direzione: {direction_accuracy:.4f}")

        # Plot
        # plt.figure(figsize=(12, 6))
        # for col in y_test.columns:
        #     plt.plot(y_test[col].values, label=f"True {col}")
        #     plt.plot(pred_df[col].values, label=f"Pred {col}", linestyle='--')
        # plt.legend()
        # plt.show()

        # --- Target classificazione ---

        X_train, X_test, y_train, y_test = train_test_split(self.X_scaled, self.y_target, test_size=0.2, shuffle=False)

        target_model = MultiOutputClassifier(
            XGBClassifier(eval_metric='logloss', n_estimators=100, max_depth=5)
        )
        target_model.fit(X_train, y_train)
        pred_target = target_model.predict(X_test)
        pred_target_df = pd.DataFrame(pred_target, columns=y_test.columns, index=y_test.index)

        print("=== Target Accuracy ===")
        for col in y_test.columns:
            acc = accuracy_score(y_test[col], pred_target_df[col])
            print(f"{col}: Accuracy={acc:.4f}")

        # --- Cumulative regressione ---
        X_train, X_test, y_train, y_test = train_test_split(self.X_scaled, self.y_cumulative, test_size=0.2, shuffle=False)

        cumulative_model = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=5))
        cumulative_model.fit(X_train, y_train)
        pred_cum = cumulative_model.predict(X_test)
        pred_cum_df = pd.DataFrame(pred_cum, columns=y_test.columns, index=y_test.index)

        print("=== Cumulative Metrics ===")
        for col in y_test.columns:
            mse = mean_squared_error(y_test[col], pred_cum_df[col])
            mae = mean_absolute_error(y_test[col], pred_cum_df[col])
            r2 = r2_score(y_test[col], pred_cum_df[col])
            print(f"{col}: MSE={mse:.6f}, MAE={mae:.6f}, R2={r2:.4f}")

        return profit_model, target_model, cumulative_model

    from itertools import combinations
    from tqdm import tqdm

    def feature_subset_search(self, max_features=None):
        """
        Itera su tutte le combinazioni possibili di features (da 1 a tutte),
        addestra i modelli e valuta le metriche.
        Usa una progress bar per tracciare l'avanzamento.
        """

        print("Inizio ricerca features subset...")
        results = []

        features = list(self.X_scaled.columns)
        if max_features is None:
            max_features = len(features)

        # Numero totale combinazioni per tqdm
        total_combinations = sum([len(list(combinations(features, r))) for r in range(1, max_features + 1)])

        with tqdm(total=total_combinations, desc="Testing feature subsets") as pbar:
            for r in range(1, max_features + 1):
                for subset in combinations(features, r):
                    X_subset = self.X_scaled[list(subset)]

                    # --- Profit regressione ---
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_subset, self.y_profit_scaled, test_size=0.2, shuffle=False
                    )

                    profit_model = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=5))
                    profit_model.fit(X_train, y_train)

                    pred = profit_model.predict(X_test)
                    pred_df = pd.DataFrame(pred, columns=y_test.columns, index=y_test.index)

                    mse_total = mean_squared_error(y_test, pred_df)
                    mae_total = mean_absolute_error(y_test, pred_df)
                    r2_total = r2_score(y_test, pred_df)

                    # Accuratezza direzione
                    direction_accuracy = np.mean(np.sign(y_test.values) == np.sign(pred_df.values))

                    # Salva i risultati
                    results.append({
                        "features": subset,
                        "mse": mse_total,
                        "mae": mae_total,
                        "r2": r2_total,
                        "direction_acc": direction_accuracy
                    })

                    pbar.update(1)

        # Ordina i risultati in base all’R2 o ad altra metrica
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by="r2", ascending=False)

        return results_df
