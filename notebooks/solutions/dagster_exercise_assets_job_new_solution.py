import pandas as pd
from dagster import asset, Config, Output, define_asset_job, Definitions, AssetSelection
from pandas import DataFrame


# This is the input data config
class DataConfig(Config):
    input_file: str = "data/genres_v2.csv"
    url: str = "https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify/data"


# This is the first Asset. Give it a proper description and name.
# Add it to an asset group named 'Data Preprocessing'
@asset(name="song_data", description="Input Music data", group_name="datapreprocessing")
def load_data(config: DataConfig) -> DataFrame:
    # TODO: test download from kaggle
    # TODO: define as resource?
    return pd.read_csv(config.input_file)


# This is the second Asset. Give it a proper description.
# Add your chosen name for the first asset as input (replace 'data').
@asset(description="Cleaned Music Data", group_name="datapreprocessing")
def data_cleaned(song_data: pd.DataFrame):
    return song_data.drop(
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


# This is the third Asset. Give it a proper description.
# Add two metadata parameter to this Asset, which tracks the number of
# rows and the number of columns. You need to define the output as
# a dagster.Output for this.
@asset(description="Duplicates to remove from Music data")
def duplicates(data_cleaned: pd.DataFrame):
    df = data_cleaned[data_cleaned.duplicated()]
    return Output(df, metadata={"num_rows": df.shape[0], "num_cols": df.shape[1]})


# This is the fourth Asset. Give it a proper description.
# Add two metadata parameter to this Asset, which tracks the number of
# rows and the number of columns. You need to define the output as
# a dagster.Output for this.
@asset(
    description="Cleaned Music Data without duplicates", group_name="datapreprocessing"
)
def data_deduplicated(data_cleaned: pd.DataFrame):
    df = data_cleaned.drop_duplicates(keep="first")
    return Output(df, metadata={"num_rows": df.shape[0], "num_cols": df.shape[1]})


# This is the fifth Asset. Give it a proper description.
# Add two metadata parameter to this Asset, which tracks the number of
# rows and the number of columns. You need to define the output as
# a dagster.Output for this.
@asset(description="Music data with One-Hot-Encoding", group_name="datapreprocessing")
def data_encoded(data_deduplicated: pd.DataFrame):
    df = pd.get_dummies(data_deduplicated, columns=["key"], prefix="key")
    return Output(df, metadata={"num_rows": df.shape[0], "num_cols": df.shape[1]})


# This is the sixth Asset. Give it a proper description.
# Add two metadata parameter to this Asset, which tracks the number of
# rows and the number of columns. You need to define the output as
# a dagster.Output for this.
# Additionally, track the standardized columns in as metadata parameters.
@asset(description="Standardized Music Data", group_name="datapreprocessing")
def data_standardized(data_encoded: pd.DataFrame):
    pd.set_option("display.max_columns", 500)
    data_encoded.describe()
    # TODO: define a Config to set which columns should be standardized
    data_encoded["duration_ms"] = (
        data_encoded["duration_ms"] - data_encoded["duration_ms"].min()
    ) / (data_encoded["duration_ms"].max() - data_encoded["duration_ms"].min())
    data_encoded["tempo"] = (data_encoded["tempo"] - data_encoded["tempo"].min()) / (
        data_encoded["tempo"].max() - data_encoded["tempo"].min()
    )
    # TODO: define output as resource
    data_encoded.to_csv("data/genres_standardized.csv", sep=";", index=False)
    return Output(
        data_encoded,
        metadata={
            "num_rows": data_encoded.shape[0],
            "num_cols": data_encoded.shape[1],
            "standardized_cols": ["duration_ms", "tempo"],
        },
    )


# You could start the UI at this point and see the assets you defined.
# Group all Assets except for 'duplicates' to the same asset group named
# 'datapreprocessing'.
# Restart the Dagster UI again and see how the assets are now grouped together.

# Define three asset jobs using 'define_asset_job'.
# 1. An Asset job, which materializes all assets.
# 2. An Asset job, which materializes all assets of the 'datapreprocessing' group.
# 3. An Asset job, which materializes only the 'duplicates' asset.
get_all_assets_job = define_asset_job(name="get_all_assets_job")
get_duplicates_job = define_asset_job(name="get_duplicates_job", selection="duplicates")
data_generation_job = define_asset_job(
    name="data_generation_job", selection=AssetSelection.groups("datapreprocessing")
)

# You will see, that the jobs are not listed in the dagster UI, yet.
# Via dagsters 'Definitions', you can define which assets and jobs should be
# visible in the dagster UI.
# Make all assets and jobs available for the dagster UI.
defs = Definitions(
    assets=[
        load_data,
        data_cleaned,
        data_deduplicated,
        duplicates,
        data_encoded,
        data_standardized,
    ],
    jobs=[get_all_assets_job, get_duplicates_job, data_generation_job],
)
