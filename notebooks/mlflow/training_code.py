import logging

import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

PATH = 'data/'


def download_data(url: str, path: str):
    data = pd.read_csv(url, sep=';')

    logger = logging.getLogger(__name__)
    logger.info(f"Data downloaded from {url}")

    data.to_csv(path, index=False)


def split_data(data_path: str):
    data = pd.read_csv(data_path)

    x = data.drop(["quality"], axis=1)

    y = data["quality"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train.to_csv(PATH + 'x_train.csv', index=False)
    y_train.to_csv(PATH + 'y_train.csv', index=False)
    x_test.to_csv(PATH + 'x_test.csv', index=False)
    y_test.to_csv(PATH + 'y_test.csv', index=False)


def train_model(x_train_path: str, y_train_path: str):
    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)

    with open('params.yaml', 'r') as fd:
        params = yaml.safe_load(fd)

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('regressor', ElasticNet(alpha=params['alpha'],
                                              l1_ratio=params['l1_ratio'],
                                              random_state=42))])

    pipe.fit(x_train, y_train)