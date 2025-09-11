from data_elaboration import DataPreprocessor, TargetGenerator
from dukascopy_python.instruments import INSTRUMENT_US_AAPL_US_USD, INSTRUMENT_US_PLTR_US_USD
from datetime import datetime, timedelta, timezone
from alpaca_service import AlpacaService
from model_factory import ModelFactory
import cols

instruments = [INSTRUMENT_US_AAPL_US_USD, INSTRUMENT_US_PLTR_US_USD]

#symbols = ["TSLA", "MSFT", "PLTR", "AAPL", "NVDA", "SPY"]

symbols = ["PLTR"]
interval = "1Day"

train_size = 0.8

total_days = 1200

now = datetime.now(timezone.utc)

start_train = (now - timedelta(days=total_days)).date()

start_test = (now - timedelta(days=total_days*train_size)).date()


def main():

    for symbol in symbols:

        print("Dowload data per:", symbol)

        alpaca = AlpacaService(symbol=symbol)

        start_train_date = datetime(start_train.year, start_train.month, start_train.day, 14, 30).strftime("%Y-%m-%dT%H:%M:%SZ")

        start_test_date = datetime(start_test.year, start_test.month, start_test.day, 14, 30).strftime("%Y-%m-%dT%H:%M:%SZ")

        end_date = datetime(now.year, now.month, now.day, 21, 0).strftime("%Y-%m-%dT%H:%M:%SZ")

        df_train = alpaca.download_data(interval, start_date=start_train_date, end_date=start_test_date )

        df_test = alpaca.download_data(interval, start_date=start_test_date, end_date=end_date)

        train_preprocessor = DataPreprocessor(df_train, symbol=symbol, interval=interval, recalculate_params=False)

        df_train_tecnical = train_preprocessor.get_dataset()

        df_train_tecnical.to_csv(f"datasets/train/{symbol}_train.csv", sep=';', encoding='utf-8')

        test_preprocessor = DataPreprocessor(df_train, symbol=symbol, interval=interval, recalculate_params=False)

        df_test_tecnical = test_preprocessor.get_dataset()

        df_test_tecnical.to_csv(f"datasets/test/{symbol}_test.csv", sep=';', encoding='utf-8')

        # preprocessor = DataPreprocessor(df_history, symbol=symbol, interval=interval)
        #
        # df = preprocessor.get_dataset()
        #
        # df_tecnical = preprocessor.generate_tecnical_dataset(df)
        #
        # target_generator = TargetGenerator()
        #
        # df_targets = target_generator.generate_all_targets_multilabel(df_tecnical)
        #
        # print("Salvataggio dataset in csv...")
        #
        # df_targets.to_csv(f"datasets/{symbol}_targets.csv", sep=';', encoding='utf-8')
        #
        # ##############################################################################
        #
        # feature_cols = cols.feature_cols
        # target_cols = cols.target_cols
        #
        # print("Esempio di utilizzo ModelFactory Semplificata")
        # print(f"Dataset shape: {df.shape}")
        # print(f"Features: {feature_cols}")
        # print(f"Targets: {target_cols}")
        #
        # # Crea ModelFactory (molto più semplice!)
        # factory = ModelFactory(
        #     df=df_targets,
        #     feature_columns=feature_cols,
        #     target_columns=target_cols
        # )
        #
        # # Addestramento diretto (senza selezione feature)
        # models, scores = factory.train_models_timeseries(n_splits=5)

        #print(scores)

        # Split per test
        # train_size = int(len(df) * 0.8)
        # df_test = df.iloc[train_size:]
        #
        # # Test predizioni
        # if len(df_test) > 0:
        #     predictions = factory.predict(df_test.head())
        #     print("\nEsempio predizioni:")
        #     print(predictions[cols.target_cols].head())
        #
        #     # Valutazione completa
        #     test_results = factory.evaluate_on_test(df_test)
        #
        # # Feature importance
        # importance_df = factory.get_feature_importance()
        # print("\nTop 10 Feature più importanti (media):")
        # avg_importance = importance_df[importance_df['target'] == 'AVERAGE'].head(10)
        # for _, row in avg_importance.iterrows():
        #     print(f"   {row['feature']}: {row['importance']:.3f}")


if __name__ == "__main__":      
    main()
