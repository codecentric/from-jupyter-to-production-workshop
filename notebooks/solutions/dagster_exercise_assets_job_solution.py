from dagster import asset, AssetExecutionContext, AssetOut, Config, job, multi_asset, op, Out, In
import pandas as pd

from notebooks.dagster.dagster_exercise_assets import download_data, split_data, \
    train_model, evaluate_model, load_params


class DownloadConfig(Config):
    url: str = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    path: str = "data/winequality-red.csv"


# This is the first Op. It's in and outputs are already defined.
@asset(
    name="data"
)
def download_data_asset(context: AssetExecutionContext, config: DownloadConfig):
    return download_data(config.url, config.path)


# Define this function as an Op and add it's in and outputs.
# ...
@multi_asset(
    outs={
        "x_train": AssetOut(),
        "y_train": AssetOut(),
        "x_test": AssetOut(),
        "y_test": AssetOut()
    }
)
def split_data_asset(context, data: pd.DataFrame):
    x_train, y_train, x_test, y_test = split_data(data)
    return x_train, y_train, x_test, y_test


# Define this function as an Op and add it's in and outputs.
# ...
@asset()
def params():
    return load_params()


# Define this function as an Op and add it's in and outputs.
# ...
@asset()
def train_model_asset(context, x_train, y_train, params):
    return train_model(x_train, y_train, params)


# Define this function as an Op and add it's in and outputs.
# ...
@asset()
def evaluate_model_asset(context, train_model_asset, x_test, y_test):
    return evaluate_model(train_model_asset, x_test, y_test)
