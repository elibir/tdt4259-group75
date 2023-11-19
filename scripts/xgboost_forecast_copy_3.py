import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import xgboost
import json
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from pathlib import Path
from pprint import pprint
import matplotlib.dates as mdates

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

    #plotting the figure
    
    plt.figure(figsize=(10, 5))
    

    #vertical lines
    #ax.axvline(y_test.index[-22], color="black", ls="--")
    #ax.axvline(y_test.index[-22-24], color="grey", ls="--")

    #plotting the data
    #train
    ax = y_train.plot(color='#ffe6a8', linewidth=2)
    train_preds.plot(ax=ax, color='#ffbb24', linewidth=2)
    
    #validation
    y_validation.plot(ax=ax, color='#c7d7ff', linewidth=2)
    validation_preds.plot(ax=ax, color='#2f6cff', linewidth=2)
    
    #predictions     
    #y_test.plot(ax=ax, color='#3ed46f', alpha=0.3, linewidth=2)
    #test_preds.plot(ax=ax, color='#3ed46f', linewidth=2) 
    
    test_preds[:-22].plot(ax=ax, color='#3ed46f', linewidth=1.8, ls="--",) 
    test_preds[-23:].plot(ax=ax, color='#3ed46f', linewidth=2)
    
    

    #ax.axvline(y_validation.index[0], ymin=0.025, ymax=0.59, color="black", ls="--", lw=1.25)
    ax.axvline(test_preds.index[0], ymin=0.025, ymax=0.59, color="black", ls="--", lw=1.25)
    ax.axvline(test_preds.index[-23], ymin=0.025, ymax=0.59, color="black", ls="--", lw=1.25)


    #ax.axvline(x=test_preds.index[-21], ymin=2 , ymax=3.5, color="black", ls="--")

    ax.legend(["Train", "Train Predictions", "Validation", "Validation Predictions", "5-day Gap Predictions", "Day-ahead Predictions"], fontsize=12)
    plt.title(f"Energy Consumption Forecast for {city}", fontsize=14)
    '''
    # x-axis
     
    #ax.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.xticks([])
    '''
    ax.set_xlabel("Time", fontsize=14)

    # Re-display the x-tick labels
    xticks = ax.get_xticks()
    xticklabels = ax.get_xticklabels()

    
    
    #ax.tick_params(axis='x', which='both', bottom=False, top=False)
    #xlim = ax.get_xlim()
    #ax.set_xlim(xlim[0], xlim[1] *1.0002)
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0]*1.0012, xlim[1]*1.0002)

    # y-axis
    ax.set_ylabel("Energy Consumption (MW)", fontsize=14)
    ax.tick_params(axis='y', which='both', labelsize=14)

    # export
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


if __name__ == "__main__":

    center_and_norm = True

    out_dir = Path("results3")
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv("../data/consumption_temp.csv")
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index(["location", "time"]).sort_index()

    metrics = {}

    for city in df.index.get_level_values("location").unique().tolist():

        #print(f"Forecasting consumption for {city}")

        #print("Finding the best hyperparams through sliding window cross validation...")

        best_params = sliding_window_evaluate(
            df.loc[city],
            max_depths=[3],  # [3, 5, 7],
            lrs=[0.1],  # [0.1, 0.01, 0.001],
            train_sizes=[500, 700, 900]
        )

        #print("Best params:")

        #print(best_params)

        #print("Training, evaluating, and plotting for the most recent data available...")

        metrics = train_and_plot_with_most_recent_data(
            df.loc[city],
            best_params["max_depth"],
            best_params["lr"],
            best_params["train_size"],
        )

        #print("Final metrics:")

        #pprint(metrics)

        with open(out_dir / f"metrics_{city}.json", "w") as mfile:
            json.dump(metrics, mfile)

        with open(out_dir / f"best_params_{city}.json", "w") as pfile:
            json.dump(best_params, pfile)
