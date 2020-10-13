from datetime import timedelta

import cv2
import json
import os

from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

PATH = "/exercise-dataset-airflow/daily/"


def preprocess_img(**kwargs):
    date_path = os.path.join(PATH, kwargs['execution_date'].strftime("%Y-%m-%d"))
    input_img = os.path.join(date_path, 'image.jpg')
    preproc_img = os.path.join(date_path, 'preprocessed.jpg')
    img = cv2.imread(input_img, cv2.IMREAD_COLOR)
    larger = cv2.resize(img, (100, 100))
    gray = cv2.cvtColor(larger, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(preproc_img, gray)


def predict_img(**kwargs):
    date_path = os.path.join(PATH, kwargs['execution_date'].strftime("%Y-%m-%d"))
    preproc_img = os.path.join(date_path, 'preprocessed.jpg')
    prediction_file = os.path.join(date_path, 'result.json')

    img = cv2.imread(preproc_img, cv2.IMREAD_GRAYSCALE)
    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, dp=2, minDist=15, param1=100, param2=70
    )
    label = "lemon" if circles is not None else "banana"
    with open(prediction_file, "w") as out:
        json.dump({"class": label}, out)

# TODO: INSERT YOUR CODE HERE