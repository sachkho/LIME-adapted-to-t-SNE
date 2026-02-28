# minhvu
# 13/11/2019
# util functions for ploting and loading dataset

import math
import joblib
import string
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine, load_iris
from sklearn.preprocessing import StandardScaler, Normalizer
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import ticker  # for customizing number of ticks


plt.rcParams.update({"font.size": 18})

def rotate_matrix(degree):
    theta = np.radians(degree)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

def load_country():
    data = joblib.load("./country.dat")
    fix_feature_names = ["Pop growth", "Pop growth 2004", "Price index", "Carbon Dioxide 2003", "Export 1990", "Export 2004", "Elec 2003", "GDP", "GDP PPP", "GDP pc", "GDP pc growth rate", "Fem Econo Rate", "Fem Econo 1990", "Fem Econo 2004", "Health Exp", "Babies", "Internet 1990", "Import 1990", "Import 2004", "Tertiary female ratio", "Babies immunized", "Manufactured Exp 2004", "Foreign invest 2004", "Military 2004", "Public Health 2003", "Private Health 2003", "Primary export 2004", "Public Health", "Refugees asylum", "Refugees origin", "Armed forces", "Parliament Seats Women", "Female Male income", "House women 2006", "Pop 1975", "Pop 2004", "Pop 2015", "Tuberculosis detected", "Tuberculosis cured 2004", "Trad fuel", "ODA pc donnor 2004", "ODA to least dev 1990", "ODA to least dev 2004", "ODA received", "ODA received pc"]
    return {
        "data": data["X"],
        "target": data["country_names"],
        "feature_names": fix_feature_names
        # clean_feature_names(data["indicator_descriptions"]),
    }

def load_automobile():
    # Ref: https://www.kaggle.com/toramky/automobile-dataset
    return joblib.load("./dataset/Automobile.pkl")

def load_cars_dataset():
    # UCI cars dataset: https://archive.ics.uci.edu/ml/datasets/automobile
    df = pd.read_csv("./dataset/cars1985.csv")
    print(df.describe())
    instance_names = df["Vehicle Name"].tolist()
    print(instance_names)
    column_names = list(df.columns)
    print(column_names)
    # The "Engine Size (l)" column is string. Convert it to float
    df["Engine Size (l)"] = df["Engine Size (l)"].str.replace(",", "").astype(float)
    # Fill NAN by the average value of the column
    df.fillna(df.mean(), inplace=True)
    # Get all columns except the first one and convert to numpy array
    data = df.loc[:, df.columns != "Vehicle Name"].to_numpy()
    print(data.shape)

    return {
        "data": data,
        "target": instance_names,
        "feature_names": column_names[1:],  # remove first column of vehicle name
    }

def load_tabular_dataset(dataset_name="country", standardize=True):
    """Load the tabular dataset with the given `dataset_name`
    Returns:
        X: [N x D] data itself
        labels: [N]
        feature_names: [D]
    """
    load_func = {
        "country": load_country,
        "cars1985": load_cars_dataset,
        "automobile": load_automobile,
        "wine": load_wine,
        "iris": load_iris,
    }[dataset_name]

    data = load_func()
    X, label_names, feature_names = (
        data["data"],
        data["target"],
        data["feature_names"],
    )
    if standardize:
        X = StandardScaler().fit_transform(X)

    return X, label_names, feature_names
