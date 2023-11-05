# MLflow Tracking CheatSheet

MLflow Tracking ist ein Framework zur Verfolgung und Organisation von Machine Learning-Experimenten. Hier ist ein CheatSheet für MLflow Tracking mit Beispielen in Python.

## 1. Installation und Importe

```bash
pip install mlflow
```

```python
import mlflow
import mlflow.sklearn  # Falls du scikit-learn für deine Modelle verwendest
```

## 2. Starten eines neuen Experiements
```python
exp_id = mlflow.create_experiment(name="exp_name")
```

## 3. Starten einer Tracking Session
```python
mlflow.start_run(experiment_id=exp_id)
```

## 4. Loggen von Parametern
```python
mlflow.log_param("param_name", param_value)
```

## 5. Loggen von Metriken
```python
mlflow.log_metric("metric_name", metric_value)
```

## 6. Loggen von Modellen
```python
model = ...  # Dein trainiertes Modell
mlflow.sklearn.log_model(model, "model_name")
```

## 7. Tags erstellen
```python
mlflow.set_tag("tag_name", "tag_value")
```

## 8. Artefakte (Bilder, JSON-Files, ...) loggen 
```python
mlflow.log_artifact("file_path_or_directory")
```
## 9. Modelle loggen
Je nach ML-Framework gibt es andere Implementierungen.
```python
mlflow.pytorch.log_model(model, "model")
```

## 10. Datasets loggen
Aus einem Numpy-Array lässt sich ein MLflow Dataset erstellen. Dieses kann mit einem Context (z.B. "Training") gespeichert werden.

```python
dataset_train = mlflow.data.from_numpy(data, source="train.csv")
mlflow.log_input(dataset_train,context="training")
```

## 11. Ending the tracking session
```python
mlflow.end_run()
```

## 12. Komplettes Beispiel
```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Starte die Tracking-Session
mlflow.start_run()

# Lade und teile die Daten in Trainings- und Testsets
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definiere und trainiere das Modell
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Protokolliere Parameter und Metriken
mlflow.log_param("n_estimators", 100)
mlflow.log_metric("accuracy", model.score(X_test, y_test))

# Protokolliere das Modell
mlflow.sklearn.log_model(model, "random_forest_model")

# Beende die Tracking-Session
mlflow.end_run()
```