import pandas as pd
from dagster import (
    asset,
    AssetOut,
    multi_asset,
    Config,
    Output,
    define_asset_job,
    Definitions,
)
from pandas import DataFrame
from sklearn.model_selection import train_test_split


class DataConfig(Config):
    input_file: str = "data/genres_v2.csv"
    url: str = "https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify/data"


class SplitConfig(Config):
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True


@asset(description="Input Music data")
def data(config: DataConfig) -> Output[DataFrame]:
    # TODO: download data instead of load?
    df = pd.read_csv(config.input_file)
    return Output(
        df,
        metadata={
            "num_rows": df.shape[0],
            "num_cols": df.shape[1],
            "source": config.url,
        },
    )


@asset(description="Cleaned Music Data")
def data_cleaned(data: pd.DataFrame):
    df = data.drop(
        [
            "type",
            "id",
            "uri",
            "track_href",
            "analysis_url",
            "song_name",
            "Unnamed: 0",
            "title",
        ],
        axis=1,
    )
    return Output(df, metadata={"num_rows": df.shape[0], "num_cols": df.shape[1]})


@asset(description="Duplicates to remove from Music data")
def duplicates(data_cleaned: pd.DataFrame):
    df = data_cleaned[data_cleaned.duplicated()]
    return Output(df, metadata={"num_rows": df.shape[0], "num_cols": df.shape[1]})


@asset(description="Cleaned Music Data without duplicates")
def data_deduplicated(data_cleaned: pd.DataFrame):
    df = data_cleaned.drop_duplicates(keep="first")
    return Output(df, metadata={"num_rows": df.shape[0], "num_cols": df.shape[1]})


@asset(description="Music data with One-Hot-Encoding")
def data_encoded(data_deduplicated: pd.DataFrame):
    df = pd.get_dummies(data_deduplicated, columns=["key"], prefix="key")
    return Output(df, metadata={"num_rows": df.shape[0], "num_cols": df.shape[1]})


@asset(description="Standardized Music Data")
def data_standardized(data_encoded: pd.DataFrame):
    pd.set_option("display.max_columns", 500)
    data_encoded.describe()
    data_encoded["duration_ms"] = (
        data_encoded["duration_ms"] - data_encoded["duration_ms"].min()
    ) / (data_encoded["duration_ms"].max() - data_encoded["duration_ms"].min())
    data_encoded["tempo"] = (data_encoded["tempo"] - data_encoded["tempo"].min()) / (
        data_encoded["tempo"].max() - data_encoded["tempo"].min()
    )
    return Output(
        data_encoded,
        metadata={"num_rows": data_encoded.shape[0], "num_cols": data_encoded.shape[1]},
    )


@multi_asset(
    description="Train-Test-Split of Music Data",
    outs={
        "x_train": AssetOut(),
        "x_test": AssetOut(),
        "y_train": AssetOut(),
        "y_test": AssetOut(),
    },
)
def split_data(data_standardized: pd.DataFrame, config: SplitConfig):
    x = data_standardized.drop(["genre"], axis=1)
    y = data_standardized["genre"]

    xtrain, xtest, ytrain, ytest = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        shuffle=config.shuffle,
    )
    return (
        Output(
            xtrain,
            metadata={
                "xtrain_num_rows": xtrain.shape[0],
                "xtrain_num_cols": xtrain.shape[1],
            },
        ),
        Output(
            xtest,
            metadata={
                "xtest_num_rows": xtest.shape[0],
                "xtest_num_cols": xtest.shape[1],
            },
        ),
        Output(
            ytrain,
            metadata={
                "ytrain_num_rows": ytrain.shape[0],
            },
        ),
        Output(
            ytest,
            metadata={
                "ytest_num_rows": ytest.shape[0],
            },
        ),
    )


# define jobs for a subset of assets
all_assets_job = define_asset_job(name="all_assets_job")
data_generation_job = define_asset_job(name="data_generation_job", selection="data")

defs = Definitions(
    assets=[
        data,
        data_cleaned,
        data_deduplicated,
        duplicates,
        data_encoded,
        data_standardized,
        split_data,
    ],
    jobs=[all_assets_job, data_generation_job],
)
