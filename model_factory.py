
import numpy as np
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_selection import SequentialFeatureSelector
from tqdm import tqdm

feature_cols = [
    "Open", "High", "Low", "Close", "Volume", "trade_count", "vwap",
    "MACD", "MACD_signal", "MACD_hist", "MACD_diff", "MACD_slope", "Signal_slope",
    "EMA_5", "EMA_10", "EMA_20", "EMA_50", "EMA_100",
    "EMA_5_x_EMA_10_diff", "EMA_5_x_EMA_10_UP_cont", "EMA_5_x_EMA_10_DOWN_cont",
    "EMA_5_x_EMA_20_diff", "EMA_5_x_EMA_20_UP_cont", "EMA_5_x_EMA_20_DOWN_cont",
    "EMA_5_x_EMA_50_diff", "EMA_5_x_EMA_50_UP_cont", "EMA_5_x_EMA_50_DOWN_cont",
    "EMA_5_x_EMA_100_diff", "EMA_5_x_EMA_100_UP_cont", "EMA_5_x_EMA_100_DOWN_cont",
    "EMA_10_x_EMA_20_diff", "EMA_10_x_EMA_20_UP_cont", "EMA_10_x_EMA_20_DOWN_cont",
    "EMA_10_x_EMA_50_diff", "EMA_10_x_EMA_50_UP_cont", "EMA_10_x_EMA_50_DOWN_cont",
    "EMA_10_x_EMA_100_diff", "EMA_10_x_EMA_100_UP_cont", "EMA_10_x_EMA_100_DOWN_cont",
    "EMA_20_x_EMA_50_diff", "EMA_20_x_EMA_50_UP_cont", "EMA_20_x_EMA_50_DOWN_cont",
    "EMA_20_x_EMA_100_diff", "EMA_20_x_EMA_100_UP_cont", "EMA_20_x_EMA_100_DOWN_cont",
    "EMA_50_x_EMA_100_diff", "EMA_50_x_EMA_100_UP_cont", "EMA_50_x_EMA_100_DOWN_cont",
    "RSI", "RSI_norm", "BB_Width", "BB_Percent_b",
    "BB_Breakout_up_cont", "BB_Breakout_down_cont",
    "Body_today", "Body_prev", "Engulfing_ratio", "Engulfing_direction", "Engulfing_strength",
    "Engulfing_inclusion", "Engulfing_index",
    "OBV", "OBV_diff", "OBV_pct", "OBV_slope", "OBV_momentum",
    "Close_VWAP_Diff", "Close_VWAP_Ratio",
    "Avg_Trade_Size", "Trade_Count_Norm_3", "Trade_Count_Norm_5", "Trade_Count_Norm_10",
    "Donchian_High", "Donchian_Low",
    "Vol_MA", "ADX", "Donchian_Close_HighDiff", "Donchian_Close_LowDiff",
    "Donchian_Close_HighPct", "Donchian_Close_LowPct",
    "Vol_Ratio", "Vol_Diff", "ADX_Above_Threshold", "ADX_Slope"
]

target_columns = [
    "Target_sl_1_1", "Target_sl_1_2", "Target_tp_1_1", "Target_tp_1_2", "Target_tp_1_3",
    "Target_sl_5_1", "Target_sl_5_2", "Target_tp_5_1", "Target_tp_5_2", "Target_tp_5_3",
    "Target_sl_10_1", "Target_sl_10_2", "Target_tp_10_1", "Target_tp_10_2", "Target_tp_10_3",
    "Target_sl_20_1", "Target_sl_20_2", "Target_tp_20_1", "Target_tp_20_2", "Target_tp_20_3"
]



class ModelFactory:
    def __init__(self, df, feature_columns=feature_cols, target_columns=target_columns, n_features=5, n_splits=3):
        """
        :param df: DataFrame contenente features e target
        :param feature_columns: lista dei nomi colonne features
        :param target_columns: lista dei nomi colonne target (es. Target_sl_1_1 ...)
        :param n_features: numero di feature da selezionare
        :param n_splits: numero di split per TimeSeriesSplit
        """
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.n_features = n_features
        self.n_splits = n_splits

        # Features
        self.X = self.df[self.feature_columns].values
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        # Target
        self.y_target = self.df[self.target_columns]

        # Dizionari per salvare risultati
        self.selected_features_dict = {}
        self.models_dict = {}
        self.scores_dict = {}


    def feature_selection(self):
        """Seleziona le migliori n_features per ciascun target separatamente, mostrando una progress bar."""
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        self.selected_features_dict = {}  # feature per target
        all_selected_features = set()  # tutte le feature selezionate almeno una volta

        for col in tqdm(self.target_columns, desc="Feature Selection Targets"):
            model = XGBClassifier(n_estimators=50, max_depth=3, eval_metric="mlogloss")
            sfs = SequentialFeatureSelector(
                model,
                n_features_to_select=self.n_features,
                direction="forward",
                scoring="accuracy",
                cv=tscv,
                n_jobs=-1
            )
            sfs.fit(self.X_scaled, self.y_target[col])
            selected_idx = sfs.get_support(indices=True)
            selected_features = [self.feature_columns[i] for i in selected_idx]

            self.selected_features_dict[col] = selected_features
            all_selected_features.update(selected_features)  # aggiunge al set globale

        print("\nFeature selezionate per ciascun target:")
        for k, v in self.selected_features_dict.items():
            print(f"{k}: {v}")

        print("\nTutte le feature selezionate almeno una volta:")
        print(sorted(all_selected_features))  # ordinato per comodità

        return self.selected_features_dict, sorted(all_selected_features)

    def train_models(self):
        """Addestra un modello XGBoost per ogni target usando le feature selezionate."""
        for col in self.target_columns:
            features = self.selected_features_dict.get(col, self.feature_columns)
            X_sel = self.df[features].values
            X_sel_scaled = self.scaler.transform(X_sel)
            model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                #subsample=0.8,  # prende solo l'80% dei dati per ogni albero
                #colsample_bytree=0.8,  # prende solo l'80% delle features per albero
                eval_metric="mlogloss"
            )
            model.fit(X_sel_scaled, self.y_target[col])
            self.models_dict[col] = model

            # Calcolo score e confusion matrix sullo stesso dataset
            y_pred = model.predict(X_sel_scaled)
            acc = accuracy_score(self.y_target[col], y_pred)
            cm = confusion_matrix(self.y_target[col], y_pred)
            self.scores_dict[col] = {"accuracy": acc, "confusion_matrix": cm}

        return self.models_dict, self.scores_dict

    def predict_current(self, new_row):
        """
        Predice i target per un nuovo campione di dati (es. ultima candela).
        new_row deve essere un dizionario {feature: valore} o una Series pandas.
        """
        predictions = {}
        for col in self.target_columns:
            features = self.selected_features_dict.get(col, self.feature_columns)
            X_new = np.array([new_row[features]])  # prendi solo le feature richieste
            X_new_scaled = self.scaler.transform(X_new)
            model = self.models_dict[col]
            y_pred = model.predict(X_new_scaled)[0]
            y_prob = model.predict_proba(X_new_scaled)[0]  # probabilità classi
            predictions[col] = {
                "prediction": int(y_pred),
                "probability": y_prob
            }
        return predictions

    def score_models(self):
        """
        Valuta tutti i modelli addestrati sui dati utilizzati per l'addestramento,
        calcola accuracy e confusion matrix per ogni target.
        """
        if not hasattr(self, "models_dict") or len(self.models_dict) == 0:
            raise ValueError("Nessun modello trovato. Addestra prima i modelli con train_models().")

        self.scores_dict = {}  # reset delle metriche

        for col in self.target_columns:
            model = self.models_dict[col]
            # seleziona le feature usate per quel target
            features = self.selected_features_dict.get(col, self.feature_columns)
            X_sel = self.df[features].values
            X_sel_scaled = self.scaler.transform(X_sel)

            # predizione
            y_pred = model.predict(X_sel_scaled)
            y_true = self.y_target[col]

            # calcolo metriche
            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)
            self.scores_dict[col] = {"accuracy": acc, "confusion_matrix": cm}

        print("Report di affidabilità dei modelli:")
        for col, metrics in self.scores_dict.items():
            print(f"\n{col} - Accuracy: {metrics['accuracy']:.3f}")
            print(metrics['confusion_matrix'])

        return self.scores_dict





