# From Jupyter to Production

## Production-ready Data Science Projects

This repository contains material for the workshop "From Jupyter to Production".
The goal of the workshop is to get a glimpse of production-readiness for data science and machine learning projects.

With the introductory Jupyter notebooks and the exercises found in the [notebooks](https://github.com/codecentric/from-jupyter-to-production-workshop/tree/master/notebooks)
directory, you will learn how to

- Versioning your data and models with DVC
- Build pipelines with Dagster
- Track experiments with MLflow
- Deploy your model with FastAPI

Having installed docker, you can use JupyterLab for the exercises.

## Start JupyterLab

First clone the repository

```bash
git clone https://github.com/codecentric/from-jupyter-to-production-workshop
cd from-jupyter-to-production-workshop
```

and then execute the command

```bash
docker compose up -d
```

You can now use JupyterLab in your browser: [http://localhost:8888](http://localhost:8888)

### Docker Images

If you want to pull the docker images separately

```bash
docker pull codecentric/from-jupyter-to-production-baseimage
```

You will find the source for the docker images here:

[http://github.com/codecentric/from-jupyter-to-production-baseimage](http://github.com/codecentric/from-jupyter-to-production-baseimage)

### Data Sources

https://archive.ics.uci.edu/ml/datasets/wine+quality
