import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xgboost
import json
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from pathlib import Path
from pprint import pprint
from xgboost import plot_importance
import seaborn as sns
import numpy as np

pd.options.mode.chained_assignment = None


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
               "preds_validation", "preds_test", "start of validation set", "start of test set"], fontsize=12)
    plt.title(f"Energy consumption forecast for {city}", fontsize=14)
    ax.set_xlabel("Time", fontsize=14) 
    ax.set_ylabel("Consumption (MW)", fontsize=14)
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.tick_params(axis='x', which='minor', labelsize=14)

    # Change font size for the numbers on the y-axis (major and minor ticks)
    ax.tick_params(axis='y', which='major', labelsize=14)
    ax.tick_params(axis='y', which='minor', labelsize=14) 
    plt.savefig(out_dir / f"{city}_forecast.png", bbox_inches='tight')
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

    # Plot the feature correlation matrix
    plot_feature_correlation(df_city, ["temperature", "dayofweek", "dayofyear", "month", "hour", "year"], out_dir, city)

    # Plot Feature Importance
    plt.figure(figsize=(8, 2))
    plt.rcParams.update({'font.size': 14})
    plt.xlabel('F Score', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plot_importance(model, height=0.5)
    plt.savefig(out_dir / f"{city}_feature_importance.png", bbox_inches='tight')
    plt.clf()
    
    
    return metrics


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

def plot_feature_correlation(df, features, out_dir, city):
    # Calculate the correlation matrix
    corr_matrix = df[features].corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Set the font size for labels and ticks
    plt.rcParams.update({'font.size': 14})

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})

    # Set the labels and title
    plt.title(f'Feature Correlation Matrix for {city}', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save the figure
    plt.savefig(out_dir / f"{city}_feature_correlation_matrix.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    center_and_norm = True

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv("../data/consumption_temp.csv")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index(["location", "time"]).sort_index()

    metrics = {}

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

        metrics = train_and_plot_with_most_recent_data(
            df.loc[city],
            best_params["max_depth"],
            best_params["lr"],
            best_params["train_size"],
        )

        print("Final metrics:")

        pprint(metrics)

        with open(out_dir / f"metrics_{city}.json", "w") as mfile:
            json.dump(metrics, mfile)

        with open(out_dir / f"best_params_{city}.json", "w") as pfile:
            json.dump(best_params, pfile)
