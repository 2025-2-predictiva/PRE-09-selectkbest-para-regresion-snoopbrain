"""Autograding script."""


def load_data():

    import pandas as pd

    dataset = pd.read_csv("files/input/auto_mpg.csv")
    dataset = dataset.dropna()
    dataset["Origin"] = dataset["Origin"].map(
        {1: "USA", 2: "Europe", 3: "Japan"},
    )
    y = dataset.pop("MPG")
    x = dataset.copy()

    return x, y


def load_estimator():

    import os
    import pickle

    if not os.path.exists("estimator.pickle"):
        return None
    with open("estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    return estimator


def test_01():

    import numpy as np

    x, y = load_data()
    estimator = load_estimator()

    y_pred = estimator.predict(x)
    correlation = np.corrcoef(y, y_pred)[0, 1]

    assert correlation > 0.9545
