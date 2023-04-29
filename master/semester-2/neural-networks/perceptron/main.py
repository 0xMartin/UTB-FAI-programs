from perceptron import *
from util import *


if __name__ == '__main__':
    # vstup od uzivatele (cesta k souboru + pomer rozdeleni)
    path = input("Zadej cestu k souboru z datasetem (perceptron_dataset.csv):")
    if len(path) == 0:
        path = 'perceptron_dataset.csv'

    ratio = input("Zadej ve relativni velikost testovaci sada (0.8):")
    if len(ratio) == 0:
        ratio = "0.8"

    # nacte dataset ze souboru a zobrazi jeho vizualizaci 3D prostoru
    df = loadDataSet(path)
    showDataset(df)

    # rozdeli dataset na testovaci a trenovaci datasety (pomer 8 : 2 + stejne zastoupeni vystupnich trid)
    splited = splitDataSet(df, float(ratio))
    print("Test dataset size =", len(splited.y_test))
    print("Train dataset size =", len(splited.y_train))

    # uceni perceptronu
    p1 = Perceptron(3)
    lerning = p1.train(splited.x_train, splited.y_train, 0.2, 10000)
    showLerningGraph(lerning)

    # testovani perceptronu
    testPerceptron(p1, splited.x_test, splited.y_test)

    # zobrazeni hyperplochy
    showDataset(df, p1)