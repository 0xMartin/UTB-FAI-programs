from ClassificationModel import *
from sklearn.neighbors import KNeighborsRegressor


class K_NearestNeighbours(ClassificationModel):
    X_train = None
    y_train = None

    def __init__(self) -> None:
        super().__init__()
        self.knnr = KNeighborsRegressor(n_neighbors=4, algorithm='auto', n_jobs=8)

    def trainModel(self, df_train) -> None:
        # prave ty veliciny jejihz prumerne hodnoty se nejvice lise mezi tridamy "Yes" a "No" (RainTomorrow)
        self.X_train = df_train.loc[:, ['Rainfall', 'Sunshine', 'RainToday', 'RISK_MM']]
        self.y_train = df_train['RainTomorrow']
        self.knnr.fit(self.X_train, self.y_train)

    def classify(self, df_test) -> CMResult:
        X_test = df_test.loc[:, ['Rainfall', 'Sunshine', 'RainToday', 'RISK_MM']]
        y_test = df_test['RainTomorrow']

        y_pred = self.knnr.predict(X_test)

        total = X_test.shape[0]
        incorrect = (y_test != y_pred).sum()
        score = 1.0 - float(incorrect) / total

        return CMResult(total_cnt=total, incorrect_cnt=incorrect, score=score)
