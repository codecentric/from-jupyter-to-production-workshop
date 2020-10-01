from datetime import datetime, timedelta

import cv2
import json
import os

from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.sensors.file_sensor import FileSensor
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(2),
}

dag = DAG(
    dag_id='file_processing',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
)

PATH = "/exercise-dataset-airflow/daily/"
DATE = "2020-10-14"
INPUT_IMG = f"/exercise-dataset-airflow/daily/{DATE}/image.jpg"

# operator that waits for the specified file to land in the file system
file_sensor = FileSensor(
    task_id="wait_data_exists",
    filepath=INPUT_IMG,
    dag=dag
)


def preprocess_img(**kwargs):
    date_path = os.path.join(PATH, kwargs['execution_date'].strftime("%Y-%m-%d"))
    input_img = os.path.join(date_path, 'image.jpg')
    preproc_img = os.path.join(date_path, 'preprocessed.jpg')
    img = cv2.imread(input_img, cv2.IMREAD_COLOR)
    larger = cv2.resize(img, (100, 100))
    gray = cv2.cvtColor(larger, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(preproc_img, gray)


# operator that executes the preprocess_img function
preprocess = PythonOperator(
    task_id="preprocess",
    python_callable=preprocess_img,
    provide_context=True,
    dag=dag,
)


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


# operator that executes the predict function      
predict = PythonOperator(
    task_id="predict",
    python_callable=predict_img,
    provide_context=True,
    dag=dag,
)

# connect our pipeline steps
preprocess >> predict
# file_sensor >> preprocess >> predict

# Test via backfilling
# airflow backfill file_processing -s 2020-09-13 -e 2020-09-16
