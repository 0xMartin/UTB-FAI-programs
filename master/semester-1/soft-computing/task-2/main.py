import pandas as pd
from prettytable import PrettyTable
from sklearn.model_selection import KFold

from NaiveBayes import *
from K_NearestNeighbours import *


def KFoldCrossValidation(k: int, df: pd.DataFrame, model: ClassificationModel):
    """
    K-Fold cross validace
    k - pocet casti, na ktery bude dataset rozdelen
    df - Vstupni dataset
    model - Klasifikacni model
    """

    df = df.replace("No", 0)
    df = df.replace("Yes", 1)
    df = df.fillna(0)

    # vysledna tabulka
    t = PrettyTable(['Fold', 'Total', 'Incorrect', 'Score'])
    cnt = 1
    avg_score = 0.0

    # k-fold cross validace
    kf = KFold(n_splits=k)
    print("Loading [", end='', flush=True)
    for train, test in kf.split(df):
        print("=", end='', flush=True)
        df_train = df.filter(items=train, axis=0)
        df_test = df.filter(items=test, axis=0)
        model.trainModel(df_train)
        res = model.classify(df_test)
        t.add_row([cnt, res.total_cnt, res.incorrect_cnt, res.score])
        avg_score += res.score
        cnt += 1
    print("]")

    # vysledky z cross validace zobrazi v konzoli
    print("%d-Fold Cross Validation" % k)
    print("Model: %s" % type(model).__name__)
    print(t)
    print("Avg score =", avg_score / k)



def printContingencyTable(df):
    df = df.replace("No", 0)
    df = df.replace("Yes", 1)
    df = df.fillna(0)

    df_yes = df[df["RainTomorrow"] == 1]
    df_no = df[df["RainTomorrow"] == 0]

    t = PrettyTable(['RainTomorrow', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
                    'Sunshine', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RISK_MM'])
    for opt, d in [("Yes", df_yes), ("No", df_no)]:
        t.add_row([opt,
                   round(d["MinTemp"].mean(), 2),
                   round(d["MaxTemp"].mean(), 2),
                   round(d["Rainfall"].mean(), 2),
                   round(d["Evaporation"].mean(), 2),
                   round(d["Sunshine"].mean(), 2),
                   round(d["Cloud9am"].mean(), 2),
                   round(d["Cloud3pm"].mean(), 2),
                   round(d["Temp9am"].mean(), 2),
                   round(d["Temp3pm"].mean(), 2),
                   round(d["RainToday"].mean(), 2),
                   round(d["RISK_MM"].mean(), 2)
                   ])
    print("[Avg values]")
    print(t)


if __name__ == '__main__':
    df = pd.read_csv('weatherAUS.csv')
    printContingencyTable(df)
    print('\n')
    KFoldCrossValidation(10, df, NaiveBayes())
    print('\n')
    KFoldCrossValidation(10, df, K_NearestNeighbours())
    print('\n')
    input("Press any key to exit ...")
