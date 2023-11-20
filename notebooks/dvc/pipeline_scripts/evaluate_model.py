import logging
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import mlflow.sklearn

PATH = 'data/'

def evaluate_model(model_path: str, x_test_path: str, y_test_path: str):
    pipe = mlflow.sklearn.load_model(model_path)

    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

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


if __name__ == "__main__":
    evaluate_model(sys.argv[1], sys.argv[2], sys.argv[3])
