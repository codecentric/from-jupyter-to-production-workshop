import sys

import pandas as pd
from sklearn.model_selection import train_test_split


PATH = 'data/'


def split_data(data_path: str):
    data = pd.read_csv(data_path)

    x = data.drop(["quality"], axis=1)

    y = data["quality"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train.to_csv(PATH + 'x_train.csv', index=False)
    y_train.to_csv(PATH + 'y_train.csv', index=False)
    x_test.to_csv(PATH + 'x_test.csv', index=False)
    y_test.to_csv(PATH + 'y_test.csv', index=False)


if __name__ == "__main__":
    split_data(sys.argv[1])
