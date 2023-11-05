import logging
from typing import Dict

import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow.sklearn

PATH = 'data/'


def download_data(url: str, path: str):
    data = pd.read_csv(url, sep=';')

    logger = logging.getLogger(__name__)
    logger.info(f"Data downloaded from {url}")

    data.to_csv(path, index=False)
    return data


def split_data(data: pd.DataFrame):

    x = data.drop(["quality"], axis=1)

    y = data["quality"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train_path = f"{PATH}x_train.csv"
    y_train_path = f"{PATH}y_train.csv"
    x_test_path = f"{PATH}x_test.csv"
    y_test_path = f"{PATH}y_test.csv"
    x_train.to_csv(x_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    x_test.to_csv(x_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    return x_train, y_train, x_test, y_test


def load_params():
    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)
    return params


def train_model(x_train, y_train, params):

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('regressor', ElasticNet(alpha=params['alpha'],
                                              l1_ratio=params['l1_ratio'],
                                              random_state=42))])

    pipe.fit(x_train, y_train)
    model_path = f"{PATH}model"
    mlflow.sklearn.save_model(pipe, model_path)
    return pipe


def evaluate_model(pipe, x_test, y_test):

    y_pred = pipe.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f.", r2)

    result = pd.DataFrame({"train": {"rmse": float(rmse),
                                     "mae": float(mae),
                                     "r2": float(r2)}})

    result.to_json(PATH + 'result.json')
    return result
