from dagster import job, op, Out, In

from notebooks.dagster.dagster_exercise_ops import download_data, split_data, \
    train_model, evaluate_model


# This is the first Op. It's in and outputs are already defined.
@op(
    ins={'path': In(str), 'url': In(str)},
    out={'data_path': Out(str)}
)
def download_data_op(context, url: str, path: str):
    data_path = download_data(url, path)
    return data_path


# Define this function as an Op and add it's in and outputs.
# ...
def split_data_op(context, data_path: str):
    x_train_path, y_train_path, x_test_path, y_test_path = split_data(data_path)
    return x_train_path, y_train_path, x_test_path, y_test_path


# Define this function as an Op and add it's in and outputs.
# ...
def train_model_op(context, x_train_path: str, y_train_path: str):
    model_path = train_model(x_train_path, y_train_path)
    return model_path


# Define this function as an Op and add it's in and outputs.
# ...
def evaluate_model_op(context, model_path: str, x_test_path: str, y_test_path: str):
    evaluate_model(model_path, x_test_path, y_test_path)


# Define this function as a job.
# ...
def data_pipeline():
    data_path = download_data_op()
    x_train_path, y_train_path, x_test_path, y_test_path = split_data_op(data_path)
    model_path = train_model_op(x_train_path, y_train_path)
    evaluate_model_op(model_path, x_test_path, y_test_path)
