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

    return df


if __name__ == "__main__":

    out_dir = Path("results2")
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(Path("data") / "consumption_temp.csv")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index(["location", "time"]).sort_index()

    for city in df.index.get_level_values("location").unique().tolist():

        print(f"Forecasting consumption for {city}")

        df_city = featurize(df.loc[city])

        features = ["temperature", "dayofweek",
                    "dayofyear", "month", "hour", "year"]
        target = "consumption"

        size = len(df_city.index)
        train_size = 300
        validation_size = 6*24
        test_size = 6*24

        train_df = df_city[size-train_size-test_size -
                           validation_size:size-test_size-validation_size]
        validation_df = df_city[size-test_size-validation_size:size-test_size]
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
        y_test[-21:].plot(ax=ax)
        train_preds.plot(ax=ax)
        validation_preds.plot(ax=ax)
        test_preds[-21:].plot(ax=ax)    
       
        ax.axvline(y_test.index[-21], color="black", ls="--")
        ax.axvline(y_test.index[-21-24], color="grey", ls="--")
        y_test[:-21].plot(ax=ax, alpha=0.2, color='green')
        test_preds[:-21].plot(ax=ax, alpha=0.2, color='brown')

        ax.legend(["y_train", "y_validation", "y_test", "preds_train",
                  "preds_validation", "preds_test", 'prediction date', 'present'])
    
        plt.savefig(out_dir / f"{city}_forecast.png")
        plt.clf()
        