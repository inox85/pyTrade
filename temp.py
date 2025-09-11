import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class ModelFactory:
    """
    Classe semplificata per classificazione multi-output con validazione temporale.
    Usa tutte le feature disponibili senza selezione.
    """

    def __init__(self, df: pd.DataFrame, feature_columns: List[str], target_columns: List[str]):
        """
        Inizializza la ModelFactory.

        Args:
            df: DataFrame contenente features e target
            feature_columns: lista dei nomi colonne features
            target_columns: lista dei nomi colonne target
        """
        self.df = df.copy()
        self.feature_columns = feature_columns
        self.target_columns = target_columns

        # Features
        self.X = self.df[self.feature_columns].values

        # Target
        self.y_target = self.df[self.target_columns]

        # Dizionari per salvare risultati
        self.models_dict = {}
        self.scores_dict = {}
        self.scalers_dict = {}  # Un scaler per ogni target

    def train_models_timeseries(self, n_splits: int = 5, xgb_params: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """
        Addestra i modelli rispettando la sequenza temporale con TimeSeriesSplit.

        Args:
            n_splits: numero di split per la validazione temporale
            xgb_params: parametri personalizzati per XGBoost

        Returns:
            Tuple con dizionari dei modelli e dei punteggi
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        self.models_dict = {}
        self.scores_dict = {}
        self.scalers_dict = {}

        # Parametri di default per XGBoost
        default_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'verbosity': 0
        }

        # Aggiorna con parametri personalizzati
        if xgb_params:
            default_params.update(xgb_params)

        print(f"Addestramento modelli con validazione temporale ({n_splits} splits)...")
        print(f"Usando {len(self.feature_columns)} features per {len(self.target_columns)} target")

        for col in tqdm(self.target_columns, desc="Training Models"):
            try:
                X_sel = self.df[self.feature_columns].values
                y_sel = self.y_target[col]

                # Verifica che il target abbia almeno 2 classi
                if len(y_sel.unique()) < 2:
                    print(f"Warning: Target {col} ha meno di 2 classi. Skipping...")
                    continue

                # Verifica che ci siano abbastanza dati
                if len(X_sel) < n_splits + 1:
                    print(f"Warning: Dati insufficienti per {col}. Skipping...")
                    continue

                accuracies = []
                cms = []

                # Cross validation temporale
                for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sel)):
                    X_train, X_test = X_sel[train_idx], X_sel[test_idx]
                    y_train, y_test = y_sel.iloc[train_idx], y_sel.iloc[test_idx]

                    # Scaling separato per ogni fold
                    fold_scaler = StandardScaler()
                    X_train_scaled = fold_scaler.fit_transform(X_train)
                    X_test_scaled = fold_scaler.transform(X_test)

                    # Modello
                    model = xgb.XGBClassifier(**default_params)
                    model.fit(X_train_scaled, y_train)

                    # Valutazione su dati futuri
                    y_pred = model.predict(X_test_scaled)
                    acc = accuracy_score(y_test, y_pred)
                    cm = confusion_matrix(y_test, y_pred)

                    accuracies.append(acc)
                    cms.append(cm)

                # Addestra il modello finale sull'intero dataset
                final_scaler = StandardScaler()
                X_scaled_full = final_scaler.fit_transform(X_sel)

                final_model = xgb.XGBClassifier(**default_params)
                final_model.fit(X_scaled_full, y_sel)

                # Salva modello e scaler
                self.models_dict[col] = final_model
                self.scalers_dict[col] = final_scaler

                # Metriche aggregate
                self.scores_dict[col] = {
                    "n_features": len(self.feature_columns),
                    "accuracy_mean": np.mean(accuracies),
                    "accuracy_std": np.std(accuracies),
                    "accuracies_all_folds": accuracies,
                    "confusion_matrix_last": cms[-1] if cms else None
                }

            except Exception as e:
                print(f"Errore nell'addestramento per {col}: {e}")
                continue

        self._print_training_report()
        return self.models_dict, self.scores_dict

    def _print_training_report(self):
        """Stampa il report di addestramento."""
        print("\n" + "=" * 60)
        print("REPORT DI ADDESTRAMENTO")
        print("=" * 60)

        for col, metrics in self.scores_dict.items():
            print(f"\n📊 {col}")
            print(f"   Features utilizzate: {metrics['n_features']}")
            print(f"   Accuracy: {metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}")

            if metrics['confusion_matrix_last'] is not None:
                print("   Ultima confusion matrix:")
                print(f"   {metrics['confusion_matrix_last']}")

    def predict(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Effettua predizioni sui nuovi dati.

        Args:
            df_new: DataFrame con i nuovi dati

        Returns:
            DataFrame con le predizioni
        """
        if not self.models_dict:
            raise ValueError("I modelli devono essere addestrati prima di fare predizioni!")

        predictions = df_new.copy()

        for col, model in self.models_dict.items():
            try:
                # Usa tutte le feature
                X_new = df_new[self.feature_columns].values

                # Scala i dati usando lo scaler del training
                X_new_scaled = self.scalers_dict[col].transform(X_new)

                # Predizione
                y_pred = model.predict(X_new_scaled)
                predictions[f"{col}_pred"] = y_pred

                # Probabilità massima
                y_pred_proba = model.predict_proba(X_new_scaled)
                predictions[f"{col}_pred_proba_max"] = np.max(y_pred_proba, axis=1)

            except Exception as e:
                print(f"Errore nella predizione per {col}: {e}")
                predictions[f"{col}_pred"] = None

        return predictions

    def predict_proba(self, df_new: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Restituisce le probabilità per tutti i target.

        Args:
            df_new: DataFrame con i nuovi dati

        Returns:
            Dizionario con le probabilità per ogni target
        """
        if not self.models_dict:
            raise ValueError("I modelli devono essere addestrati prima di fare predizioni!")

        probabilities = {}

        for col, model in self.models_dict.items():
            try:
                X_new = df_new[self.feature_columns].values
                X_new_scaled = self.scalers_dict[col].transform(X_new)

                probabilities[col] = model.predict_proba(X_new_scaled)

            except Exception as e:
                print(f"Errore nella predizione delle probabilità per {col}: {e}")
                probabilities[col] = None

        return probabilities

    def evaluate_on_test(self, df_test: pd.DataFrame) -> Dict:
        """
        Valuta i modelli su un set di test.

        Args:
            df_test: DataFrame di test con i target reali

        Returns:
            Dizionario con le metriche di valutazione
        """
        if not self.models_dict:
            raise ValueError("I modelli devono essere addestrati prima della valutazione!")

        predictions = self.predict(df_test)
        results = {}

        print("\n" + "=" * 60)
        print("VALUTAZIONE SU TEST SET")
        print("=" * 60)

        for col in self.target_columns:
            if col in self.models_dict:
                y_true = df_test[col]
                y_pred = predictions[f"{col}_pred"]

                # Rimuovi eventuali NaN
                mask = ~(pd.isna(y_true) | pd.isna(y_pred))
                y_true_clean = y_true[mask]
                y_pred_clean = y_pred[mask]

                if len(y_true_clean) > 0:
                    accuracy = accuracy_score(y_true_clean, y_pred_clean)
                    cm = confusion_matrix(y_true_clean, y_pred_clean)
                    report = classification_report(y_true_clean, y_pred_clean, output_dict=True)

                    results[col] = {
                        "accuracy": accuracy,
                        "confusion_matrix": cm,
                        "classification_report": report
                    }

                    print(f"\n📊 {col}")
                    print(f"   Test Accuracy: {accuracy:.3f}")
                    print(f"   Confusion Matrix:\n{cm}")

        return results

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Restituisce l'importanza delle feature per tutti i modelli.

        Returns:
            DataFrame con l'importanza delle feature
        """
        if not self.models_dict:
            raise ValueError("I modelli devono essere addestrati prima di ottenere l'importanza!")

        importance_data = []

        for target, model in self.models_dict.items():
            importances = model.feature_importances_

            for feature, importance in zip(self.feature_columns, importances):
                importance_data.append({
                    'target': target,
                    'feature': feature,
                    'importance': importance
                })

        importance_df = pd.DataFrame(importance_data)

        # Aggiungi anche l'importanza media tra tutti i target
        avg_importance = importance_df.groupby('feature')['importance'].mean().reset_index()
        avg_importance['target'] = 'AVERAGE'

        return pd.concat([importance_df, avg_importance], ignore_index=True).sort_values('importance', ascending=False)