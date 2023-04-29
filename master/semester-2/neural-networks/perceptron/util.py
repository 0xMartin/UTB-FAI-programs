import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from perceptron import *


def loadDataSet(file_name) -> pd.DataFrame:
    """
    file_name -> nazev souboru kde se nachazi dataset
    """

    return pd.read_csv(file_name)


class SplitedDataset:
    x_train: list = None
    y_train: list = None
    x_test: list = None
    y_test: list = None
    def __init__(self, x_train, y_train, x_test, y_test) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


def splitDataSet(df: pd.DataFrame, ratio: float) -> SplitedDataset:
    """
    df -> pandas dataframe
    ratio -> pomer rozdeleni datasetu na testovaci a trenovaci dataset (ratio -> relativni velikost testovaciho datasetu)
    """

    split = StratifiedShuffleSplit(n_splits=1, test_size=(1-ratio), random_state=42)

    for train_index, test_index in split.split(df, df["Occupancy"]):
        train = df.iloc[train_index] 
        test = df.iloc[test_index]
    
    x_train = train.drop("Occupancy", axis=1).values
    y_train = train["Occupancy"].values

    x_test = test.drop("Occupancy", axis=1).values
    y_test = test["Occupancy"].values

    return SplitedDataset(x_train, y_train, x_test, y_test)    


def showDataset(df: pd.DataFrame, perceptron: Perceptron = None):
    """
    df -> pandas dataframe
    perceptron -> perceptorn, referenci predavat jen pokud chceme zobrazi hyperplochu rozdelujici prostor
    """

    # vytvoření 3D bodového grafu
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # zobrazit body s různými barvami v závislosti na hodnotě sloupce "Occupancy"
    # Occupancy == 0
    list_tmp = list()
    list_light = list()
    list_co2 = list()
    for _, row in df.iterrows():
        if row["Occupancy"] == 0:
            list_tmp.append(row["Temperature"])
            list_light.append(row["Light"])
            list_co2.append(row["CO2"])
    ax.scatter(list_tmp, list_light, list_co2, c="red", label="0")
    # Occupancy == 1
    list_tmp.clear()
    list_light.clear()
    list_co2.clear()
    for _, row in df.iterrows():
        if row["Occupancy"] == 1:
            list_tmp.append(row["Temperature"])
            list_light.append(row["Light"])
            list_co2.append(row["CO2"])
    ax.scatter(list_tmp, list_light, list_co2, c="blue", label="1")

    # zobrazi hyperplochu perceptronu rozdelujici prostor
    if perceptron != None:
        x, y = np.meshgrid(np.linspace(df['Temperature'].min(), df['Temperature'].max(), 2),
                            np.linspace(df['Light'].min(), df['Light'].max(), 2))
        w = perceptron.w_list
        z = -(w[0] * x + w[1] * y) / w[2]
        ax.plot_surface(x, y, z, alpha=0.6)

    # nastavení popisků osy X, Y, Z
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Light")
    ax.set_zlabel("CO2")
    ax.legend()

    # zobrazení grafu s legendou
    plt.show()


def showLerningGraph(lerning: list):
    epochs = range(1, len(lerning) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lerning)
    plt.xlabel('Epocha')
    plt.ylabel('Pocet chyb')
    plt.show()


def testPerceptron(p: Perceptron, test_x: list, test_y: list):
    """
    p - perceptron
    test_x - vstupni hodnoty testovaci sady
    test_y - vystupni hodnoty testovaci sady
    """
    correct = 0
    wrong = 0

    for i in range(len(test_x)):
        y = p.predict(test_x[i])
        if y == test_y[i]:
            correct += 1
        else:
            wrong += 1
      
    print("\n[TEST]") 
    print("Correct =", correct, ",", correct / len(test_x) * 100, "%")  
    print("Wrong =", wrong, ",", wrong / len(test_x) * 100, "%") 