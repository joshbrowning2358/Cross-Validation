from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from cross_validation import CrossValidation

d = pd.DataFrame({'x': range(100)})
d['y'] = d['x'] + np.random.rand(100)*10


def mse(x, y):
    return np.mean([(x_val-y_val)**2 for x_val, y_val in zip(x, y)])

cv = CrossValidation(d, ['x'], 'y', mse)
cv.run_cv(Lasso())