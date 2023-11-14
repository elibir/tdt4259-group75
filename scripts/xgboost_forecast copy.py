import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xgboost
from pathlib import Path


matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
matplotlib.rcParams['figure.figsize'] = (20, 10)


def featurize(df: pd.DataFrame):
    df["dayofweek"] = df.index.dayofweek
    df["dayofyear"] = df.index.dayofyear
    df["month"] = df.index.month
    df["hour"] = df.index.hour
    df["year"] = df.index.year
    # df["lag_7d"] = df["consumption"].shift(24*7)
    return df

if __name__ == "__main__":

    out_dir = Path("results1")
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(Path("data") / "consumption_temp.csv")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index(["location", "time"]).sort_index()

    for city in df.index.get_level_values("location").unique().tolist():

        print(f"Forecasting consumption for {city}")

        df_city = featurize(df.loc[city])

        features = ["temperature", "dayofweek",
                    "dayofyear", "month", "hour", "year", 
                    # "lag_7d"
                    ]
        target = "consumption"

        size = len(df_city.index)
        train_size = 300
        validation_size = 6*24
        test_size = 1*21

        train_df = df_city[size-train_size-test_size - (24*5) -
                           validation_size:size-test_size - (24*5) -validation_size]
        validation_df = df_city[size-test_size-(24*5)-validation_size:size-test_size-(24*5)]
        test_df = df_city[size-test_size:]

        X_train, y_train = train_df[features], train_df[target]
        X_validation, y_validation = validation_df[features], validation_df[target]
        X_test, y_test = test_df[features], test_df[target]

        model = xgboost.XGBRegressor(base_score=0.5, booster="gbtree",
                                     n_estimators=1000,
                                     early_stopping_rounds=50,
                                     objective="reg:squarederror",
                                     max_depth=3,
                                     learning_rate=0.01)

        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_validation, y_validation)],
                  verbose=100)

        train_preds = pd.DataFrame(
            index=X_train.index, data=model.predict(X_train))
        validation_preds = pd.DataFrame(
            index=X_validation.index, data=model.predict(X_validation))
        test_preds = pd.DataFrame(
            index=X_test.index, data=model.predict(X_test)
        )

        ax = y_train.plot()
        y_validation.plot(ax=ax)
        y_test.plot(ax=ax)
        train_preds.plot(ax=ax)
        validation_preds.plot(ax=ax)
        test_preds.plot(ax=ax)

        ax.legend(["y_train", "y_validation", "y_test", "preds_train",
                  "preds_validation", "preds_test"])
        
        plt.savefig(out_dir / f"{city}_forecast.png")
        plt.clf()
        
        ax = y_test.plot(color='green')
        test_preds.plot(ax=ax, color='brown')       

        ax.legend(["y_test", "preds_test"])
        
        plt.savefig(out_dir / f"{city}_forecast_daily.png")
        plt.clf()
