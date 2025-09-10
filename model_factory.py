import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

class ModelFactory:
    def __init__(self, df, train_col, target_col):
        """
        ModelFactory focalizzata sulla classificazione multi-classe TP/SL.

        Args:
            df (pd.DataFrame): dataset completo
            train_col (list): colonne delle features
            target_col (list): colonne target (multi-classe TP/SL)
        """

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

        self.X_scaled.to_csv("datasets/X_scaled.csv", sep=";", encoding="utf-8")

        # ====== Target classificazione ======
        self.target_columns = target_col
        self.y_target = self.df_train[target_col]

        # ====== Ricerca subset di feature ======
        best_results_global, best_results_per_horizon = self.feature_subset_search(max_features=2)

        print("======================")
        print("Ranking globale:\n", best_results_global)
        print("======================")
        print("Ranking per orizzonte:\n", best_results_per_horizon)
        print("======================")

        # ====== Generazione modelli ======
        self.models = self.generate_models()

    def feature_subset_search(self, max_features=2):
        """
        TODO: implementa la logica di ricerca feature migliori.
        Deve restituire due ranking:
        - globale
        - per orizzonte temporale
        """
        # mock temporaneo
        best_results_global = {"feat1": 0.85, "feat2": 0.80}
        best_results_per_horizon = {
            "H_10": {"feat1": 0.87, "feat3": 0.75},
            "H_20": {"feat2": 0.83, "feat4": 0.78}
        }
        return best_results_global, best_results_per_horizon

    def generate_models(self, test_size=0.2, random_state=42):
        """
        Genera modelli multi-output per classificazione multi-target.

        Returns:
            dict: modelli addestrati per ogni target
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y_target, test_size=test_size, random_state=random_state
        )

        model = MultiOutputClassifier(XGBClassifier(
            n_estimators=100,
            max_depth=5,
            use_label_encoder=False,
            eval_metric="mlogloss"
        ))

        model.fit(X_train, y_train)

        print("Modelli multi-output addestrati con successo!")
        return model
