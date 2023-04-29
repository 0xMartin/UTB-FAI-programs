from ClassificationModel import *
from sklearn.naive_bayes import GaussianNB


class NaiveBayes(ClassificationModel):
    X_train = None
    y_train = None

    def __init__(self) -> None:
        super().__init__()
        self.gnb = GaussianNB()

    def trainModel(self, df_train) -> None:
        self.X_train = df_train.loc[:, ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
                                        'Sunshine', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RISK_MM']]
        self.y_train = df_train['RainTomorrow']
        self.gnb.fit(self.X_train, self.y_train)

    def classify(self, df_test) -> CMResult:
        X_test = df_test.loc[:, ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation',
                                 'Sunshine', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RISK_MM']]
        y_test = df_test['RainTomorrow']

        y_pred = self.gnb.predict(X_test)

        total = X_test.shape[0]
        incorrect = (y_test != y_pred).sum()
        score = 1.0 - float(incorrect) / total

        return CMResult(total_cnt=total, incorrect_cnt=incorrect, score=score)
