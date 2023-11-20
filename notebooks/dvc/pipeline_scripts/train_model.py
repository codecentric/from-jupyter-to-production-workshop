import sys

import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline

import mlflow.sklearn

PATH = 'data/'

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

    mlflow.sklearn.save_model(pipe, PATH + "model")

if __name__ == "__main__":
    train_model(sys.argv[1], sys.argv[2])
