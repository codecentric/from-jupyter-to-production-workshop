# From Jupyter to Production
## Data Science Production Workflow

This repository contains material for the workshop "From Jupyter to Production".
The goal of the workshop is to get a glimpse of production-readiness for data science and machine learning projects.

With the introductory Jupyter notebooks and the exercises found in the [notebooks](https://github.com/codecentric/from-jupyter-to-production-workshop/tree/master/notebooks)
directory, you will learn how to

* Versioning your data and models with DVC
* Build pipelines with Kedro
* Track Experiments with MLflow
* Deploy your model with FastAPI
* Orchestration with Airflow

## Docker Images
```bash
docker pull radtkem/from-jupyter-to-production-baseimage
docker pull radtkem/from-jupyter-to-production-airflow-image
```

## Start JupyterLab
```bash
docker-compose up -d
```

### Data Sources

https://www.kaggle.com/moltean/fruits

https://archive.ics.uci.edu/ml/datasets/wine+quality
