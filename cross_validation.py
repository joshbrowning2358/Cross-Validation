import pandas as pd
import numpy as np


class CrossValidation:
    """
    Class to perform automatic cross-validation when given data and a
    sampling approach.
    """

    def __init__(self, data, X_columns, y_column, metric, strategy={'type': 'shuffle', 'fold_cnt': 10}):
        """
        :param data: A pandas object containing the training data.
        :param X_columns: column names of data corresponding to the independent variables.
        :param y_column: column name of data corresponding to the dependent variables.
        :param metric: A function accepting two vectors and computing an error metric.
        :param strategy: The cross-validation 'strategy'.  Namely, this should
        be a dictionary providing all the configuration parameters for CV.
        Currently supported structures:
        {'type': shuffle, 'fold_cnt': numeric (optional)}
        """
        self.data = data
        self.X_columns = X_columns
        self.y_column = y_column
        self.strategy = strategy
        self.metric = metric
        self._check_input_types()

    def _check_input_types(self):
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError('data must be of type DataFrame!')
        if not isinstance(self.strategy, dict):
            raise TypeError('strategy must be of type dict!')
        if 'fold' in self.data.columns:
            raise NameError("data cannot have a column named 'fold'!")
        if not all([x in self.data.columns for x in self.X_columns]):
            raise NameError("not all X_columns are valid column names of data!")
        if self.y_column not in self.data.columns:
            raise NameError("y_column is not a valid column name of data!")

    def run_cv(self, models):
        """
        :param models: A list of instances with fit and predict methods.
        """
        if not isinstance(models, list):
            models = [models]
        self._assign_cv_fold()
        error_metric = pd.DataFrame({'model': [], 'fold': [], 'error': []})
        for m in models:
            for fold_number in self.data.fold.unique():
                error = self._run_cv_fold(m, fold_number)
                error_metric = error_metric.append({'model': m, 'fold': fold_number, 'error': error}, ignore_index=True)
        return error_metric

    def _assign_cv_fold(self):
        if self.strategy['type'] == 'shuffle':
            records_per_fold = np.ceil(self.data.shape[0] / self.strategy['fold_cnt'])
            fold_numbers = np.random.permutation(range(self.strategy['fold_cnt'])*records_per_fold)
            self.data['fold'] = fold_numbers[:self.data.shape[0]]
        else:
            raise ValueError("Input strategy['type'] is not yet implemented!")

    def _run_cv_fold(self, m, fold_number):
        train_filter = self.data['fold'] == fold_number
        cv_filter = [not x for x in train_filter]
        m.fit(X=self.data.ix[train_filter, self.X_columns], y=self.data.ix[train_filter, self.y_column])
        prediction = m.predict(X=self.data.ix[cv_filter, self.X_columns])
        return self.metric(prediction, self.data.ix[cv_filter, self.y_column])
