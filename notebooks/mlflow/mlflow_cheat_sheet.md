# 🚀 MLflow Tracking CheatSheet 🚀

MLflow Tracking is a framework for tracking and organizing machine learning experiments. Here's a CheatSheet for MLflow Tracking with Python examples.

## 1. Installation and Imports

```bash
pip install mlflow
```
```python
import mlflow  # Import the main MLflow library
import mlflow.sklearn  # If using scikit-learn for your models
```

## 2. Starting a New Experiment
Create a new MLflow experiment with a specified name.
```python
exp_id = mlflow.create_experiment(name="exp_name")
```

## 3. Starting a Tracking Session
Start a new MLflow run within the specified experiment.
```python
mlflow.start_run(experiment_id=exp_id)  
```

## 4. Logging Parameters
Log a parameter and its value for the current run.
```python
mlflow.log_param("param_name", param_value)  
```
## 5. Logging Metrics
Log a metric and its value for the current run.
```python
mlflow.log_metric("metric_name", metric_value)  
```
## 6. Logging Models
Log the trained model with a specified name.
Depending on the ML framework, there are different implementations. For scikit-learn models, you can use `mlflow.sklearn.log_model()`.
```python
model = ...  # Your trained sklearn model. Other model flavors are available
mlflow.sklearn.log_model(model, "model_name")
```
## 7. Creating Tags
Add a tag to the current run for additional metadata.
```python
mlflow.set_tag("tag_name", "tag_value") 
```

## 8. Logging Artifacts (Images, JSON Files, etc.)
Log an artifact (file or directory) associated with the current run.
```python
mlflow.log_artifact("file_path_or_directory")
```
## 9. Logging Datasets
You can create an MLflow Dataset from a NumPy array. This can be saved with a context (e.g., "Training").
```python
dataset_train = mlflow.data.from_numpy(data, source="train.csv")  # Create a dataset from a NumPy array
mlflow.log_input(dataset_train, context="training")  # Log the dataset with a specified context
```
## 10. Ending the Tracking Session
End the current run and flush all logged data to the backend store.
```python
mlflow.end_run()
```

## 11. Complete Example
```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Start the tracking session
mlflow.start_run()

# Load and split the data into training and test sets
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Log parameters and metrics
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("accuracy", model.score(X_test, y_test))

# Log the model
mlflow.sklearn.log_model(model, "random_forest_model")

# End the tracking session
mlflow.end_run()
```