import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Lasso

from cross_validation import CrossValidation


class MyLasso:

    def __init__(self):
        self.model = Lasso(alpha=0)

    def fit(self, X, y):
        X = self.filter_X(X)
        self.model.fit(X, y)

    def predict(self, X):
        X = self.filter_X(X)
        return self.model.predict(X)

    @staticmethod
    def filter_X(X):
        # pass list so a DataFrame is returned
        return X.loc[:, ['age']]


def my_roc(actual, predicted):
    actual = [int(x) for x in actual]
    return roc_auc_score(actual, predicted)

train = pd.read_csv('gender_age_train.csv')
train['train_fl'] = True
train['gender'] = [float(x == 'M') for x in train['gender']]
test = pd.read_csv('gender_age_test.csv')
test['gender'] = ''
test['age'] = -1
test['group'] = ''
test['train_fl'] = False
data = pd.concat([train, test])

cv = CrossValidation(data, my_roc, 'gender', 'train_fl', id_col='device_id', logged=True)
model = MyLasso()
cv.run(model, 'test_age_gender')
