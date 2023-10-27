import os

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


seed = 42
# generate data
X, y = make_classification(n_samples = 1000, random_state=seed)

# split into train / test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

# Save it
os.makedirs("data", exist_ok=True)
np.savetxt("data/train_features.csv", X_train)
np.savetxt("data/test_features.csv", X_test)
np.savetxt("data/train_labels.csv", y_train)
np.savetxt("data/test_labels.csv", y_test)