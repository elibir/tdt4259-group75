import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xgboost
import json
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from pathlib import Path
from pprint import pprint
import numpy as np

pd.options.mode.chained_assignment = None


matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'
matplotlib.rcParams['figure.figsize'] = (20, 10)

error_dir = Path("error")
error_dir.mkdir(exist_ok=True)

def featurize(df: pd.DataFrame):
    df["dayofweek"] = df.index.dayofweek
    df["dayofyear"] = df.index.dayofyear
    df["month"] = df.index.month
    df["hour"] = df.index.hour
    df["year"] = df.index.year

    return df


def train_and_plot_with_most_recent_data(
        df_city,
        max_depth,
        lr,
        train_size,
        validation_size=6*24,
        test_size=6*24-2): #Subtract two hours to ensure test and validation sets start at midnight

    df_city = featurize(df_city)

    features = ["temperature", "dayofweek",
                "dayofyear", "month", "hour", "year"]
    target = "consumption"

    size = len(df_city.index)

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
                                 max_depth=max_depth,
                                 learning_rate=lr)

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
    ax.axvline(y_validation.index[0], color="grey", ls="--")
    ax.axvline(y_test.index[0], color="black", ls="--")

    ax.legend(["y_train", "y_validation", "y_test", "preds_train",
               "preds_validation", "preds_test", "start of validation set", "start of test set"])
    plt.title(f"Energy consumption forecast for {city}")
    plt.savefig(out_dir / f"{city}_forecast.png")
    plt.clf()

    metrics = {
        "train": {
            "mae": mean_absolute_error(y_train, train_preds),
            "rmse": mean_squared_error(y_train, train_preds) ** (1/2),
            "mape": mean_absolute_percentage_error(y_train, train_preds)
        },
        "validation": {
            "mae": mean_absolute_error(y_validation, validation_preds),
            "rmse": mean_squared_error(y_validation, validation_preds) ** (1/2),
            "mape": mean_absolute_percentage_error(y_validation, validation_preds)
        },
        "test": {
            "mae": mean_absolute_error(y_test, test_preds),
            "rmse": mean_squared_error(y_test, test_preds) ** (1/2),
            "mape": mean_absolute_percentage_error(y_test, test_preds)
        }
    }

    # Convert metrics to a JSON-serializable format
    metrics = {
        k: v.item() if isinstance(v, np.generic) else v
        for k, v in metrics.items()
    }

    error = y_test - test_preds.iloc[:, 0]  # Calculate the error
    error_df = error.to_frame('error')  # Convert the series to a DataFrame for CSV
    error_dir = Path("error")
    error_dir.mkdir(exist_ok=True)
    error_csv_path = error_dir / f"error_{city}.csv"  # Define the error CSV path
    error_df.reset_index().to_csv(error_csv_path, index=False)  # Save the error to a CSV file

    return metrics, error  # Return the error along with the metrics


def train_and_evaluate(train_df, validation_df, max_depth, lr, plot=False):

    train_df, validation_df = featurize(train_df), featurize(validation_df)

    features = ["temperature", "dayofweek",
                "dayofyear", "month", "hour", "year"]
    target = "consumption"

    X_train, y_train = train_df[features], train_df[target]
    X_validation, y_validation = validation_df[features], validation_df[target]

    model = xgboost.XGBRegressor(base_score=0.5, booster="gbtree",
                                 n_estimators=1000,
                                 early_stopping_rounds=50,
                                 objective="reg:squarederror",
                                 max_depth=max_depth,
                                 learning_rate=lr,
                                 )

    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_validation, y_validation)], verbose=0)

    train_preds = pd.DataFrame(
        index=X_train.index, data=model.predict(X_train))
    validation_preds = pd.DataFrame(
        index=X_validation.index, data=model.predict(X_validation))

    metrics = {
        "train": {
            "mae": mean_absolute_error(y_train, train_preds),
            "rmse": mean_squared_error(y_train, train_preds) ** (1/2),
            "mape": mean_absolute_percentage_error(y_train, train_preds)
        },
        "validation": {
            "mae": mean_absolute_error(y_validation, validation_preds),
            "rmse": mean_squared_error(y_validation, validation_preds) ** (1/2),
            "mape": mean_absolute_percentage_error(y_validation, validation_preds)
        },
    }
    

    return metrics


def sliding_window_evaluate(
        df_city,
        max_depths,
        lrs,
        train_sizes,
        target="rmse",
        validation_size=6*24,
        test_size=6*24,
        data_start_percentage=0.9
):
    """
    Function for sliding window cross validation. 
    The validation and train sets are time shifted, starting
    at data_start_percentage * total data points, and ending just before 
    the validation set starts overlapping with the final test set (most recent data).
    The total loss for a set of hyperparams is the average of losses over the different 
    time intervals. 
    The function tries all hyperparam values supplied, and returns the best ones.
    """

    size = len(df_city.index)

    min_avg_val_loss = float("inf")
    top_params = {
        "max_depth": max_depths[0],
        "lr": lrs[0],
        "train_size": train_sizes[0]
    }

    progress_bar = tqdm(total=len(max_depths)*len(lrs)*len(train_sizes))

    for md in max_depths:
        for lr in lrs:
            for train_size in train_sizes:

                avg_valid_loss = 0
                data_start = int(data_start_percentage * size)

                while data_start < size - test_size - validation_size - train_size:

                    train_df = df_city.iloc[data_start:data_start+train_size]
                    valid_df = df_city.iloc[data_start +
                                            train_size:data_start+train_size+validation_size]

                    metrics = train_and_evaluate(train_df, valid_df, md, lr)
                    avg_valid_loss += metrics["validation"][target]
                    data_start += train_size

                avg_valid_loss /= (size - test_size -
                                   validation_size) // train_size

                if avg_valid_loss < min_avg_val_loss:
                    min_avg_val_loss = avg_valid_loss
                    top_params = {
                        "max_depth": md,
                        "lr": lr,
                        "train_size": train_size
                    }

                progress_bar.update(1)

    return top_params


if __name__ == "__main__":

    center_and_norm = True

    out_dir = Path("error")
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv("../data/consumption_temp.csv")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index(["location", "time"]).sort_index()

    metrics = {}
    errors = {}

    for city in df.index.get_level_values("location").unique().tolist():
        print(f"Forecasting consumption for {city}")
        print("Finding the best hyperparams through sliding window cross validation...")

        best_params = sliding_window_evaluate(
            df.loc[city],
            max_depths=[3],  # [3, 5, 7],
            lrs=[0.1],  # [0.1, 0.01, 0.001],
            train_sizes=[500, 700, 900]
        )

        print("Best params:")
        print(best_params)

        print("Training, evaluating, and plotting for the most recent data available...")

        metrics, error = train_and_plot_with_most_recent_data(
            df.loc[city],
            best_params["max_depth"],
            best_params["lr"],
            best_params["train_size"],
        )

        errors[city] = error  # Store the error for plotting

        print("Final metrics:")
        print(metrics)

        # Save metrics to JSON file
        #metrics_json_path = out_dir / f"metrics_{city}.json"
        #with open(metrics_json_path, "w") as mfile:
        #    json.dump(metrics, mfile)

        # Save the best parameters to JSON file
        #best_params_json_path = out_dir / f"best_params_{city}.json"
        #with open(best_params_json_path, "w") as pfile:
        #    json.dump(best_params, pfile)

    plt.figure(figsize=(12, 8))

    for city, error in errors.items():
        time_index = error.index.to_pydatetime()
        plt.plot(time_index, error.values, label=f"Error for {city}")

    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.title("Error Comparison Among Cities", y=1.05)
    plt.ylabel("Error (Actual - Predicted)")
    plt.xlabel("Time")
    plt.savefig(error_dir / "error_comparison.png", bbox_inches='tight')  # Save to 'error' directory
    plt.clf()