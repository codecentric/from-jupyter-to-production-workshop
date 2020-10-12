# From Jupyter to Production
## Production-ready Data Science Projects

This repository contains material for the workshop "From Jupyter to Production".
The goal of the workshop is to get a glimpse of production-readiness for data science and machine learning projects.

With the introductory Jupyter notebooks and the exercises found in the [notebooks](https://github.com/codecentric/from-jupyter-to-production-workshop/tree/master/notebooks)
directory, you will learn how to

* Versioning your data and models with DVC
* Build pipelines with Kedro
* Track Experiments with MLflow
* Deploy your model with FastAPI
* Orchestration with Airflow

Having installed docker and docker-compose, you can use JupyterLab for the exercises. 
## Start JupyterLab
First clone the repository
```bash
git clone https://github.com/codecentric/from-jupyter-to-production-workshop
cd from-jupyter-to-production-workshop
```
and then execute the command
```bash
docker-compose up -d
```
You can now use JupyterLab in your browser: [http://localhost:8888](http://localhost:8888)

The Airlow UI can be accessed via: [http://localhost:8080](http://localhost:8080)

### Docker Images
```bash
docker pull radtkem/from-jupyter-to-production-baseimage
docker pull radtkem/from-jupyter-to-production-airflow-image
```

### Data Sources

https://www.kaggle.com/moltean/fruits

https://archive.ics.uci.edu/ml/datasets/wine+quality
