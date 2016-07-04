import pandas as pd
import numpy as np
import re
import csv


class CrossValidation:
    """
    Class to perform automatic cross-validation when given data and a
    sampling approach.
    """

    def __init__(self, train, test, X_columns, y_column, metric, cv_index_col=None,
                 strategy=None):
        """
        :param train: A pandas object containing the training data.
        :param X_columns: column names of train/test corresponding to the independent variables.
        :param y_column: column name of train corresponding to the dependent variables.
        :param metric: A function accepting two vectors and computing an error metric.
        :param cv_index_col: column name of train indicating which values should be used
        for training and which for validation (all are used for the final test).  If only two
        unique values are available, the first value will be assumed to correspond to the training
        set and the second the validation set (i.e. no cross-validation will be performed).
        :param strategy: The cross-validation 'strategy'.  Namely, this should
        be a dictionary providing all the configuration parameters for CV.
        Currently supported structures:
        {'type': shuffle, 'fold_cnt': numeric (optional)}
        """
        self.train = train
        self.test = test
        self.X_columns = X_columns
        self.y_column = y_column
        self.strategy = strategy
        self.metric = metric
        self.cv_index_col = cv_index_col
        self._check_input_types()

    def _check_input_types(self):
        if not isinstance(self.train, pd.DataFrame):
            raise TypeError('train must be of type DataFrame!')
        if not isinstance(self.test, pd.DataFrame):
            raise TypeError('test must be of type DataFrame!')
        if self.cv_index_col is None and not isinstance(self.strategy, dict):
            raise TypeError('strategy must be of type dict if cv_index_col is missing!')
        if self.strategy is None and not isinstance(self.cv_index_col, str):
            raise TypeError('cv_index_col must be of type str if strategy is missing!')
        if 'fold' in self.train.columns:
            raise NameError("train cannot have a column named 'fold'!")
        if not all([x in self.train.columns for x in self.X_columns]):
            raise NameError("not all X_columns are valid column names of train!")
        if not all([x in self.test.columns for x in self.X_columns]):
            raise NameError("not all X_columns are valid column names of test!")
        if not self.cv_index_col in self.train.columns:
            raise NameError("cv_index_col is not a valid column name of train!")
        if self.y_column not in self.train.columns:
            raise NameError("y_column is not a valid column name of train!")

    def run(self, models, filename):
        """
        :param models: A list of instances with fit and predict methods.
        :param filename: Str, used to write out result files (from cv or train/test)
        """
        if not isinstance(models, list):
            models = [models]
        model_key = np.random.randint(100000000000)
        self._run_cv(self, models, filename, model_key)
        self._run_final(self, models, filename, model_key)
        # self._log_results(models, filename, model_key)

    def _run_cv(self, models, filename, model_key):
        if self.cv_index_col is None:
            self._assign_cv_fold()
        error_metric = pd.DataFrame({'model': [], 'fold': [], 'error': []})
        for m in models:
            folds_to_run = self.train.fold.unique()
            cv_prediction = np.repeat(np.nan, len(self.train))
            if len(folds_to_run) == 2:
                folds_to_run = np.min(folds_to_run)
            for fold_number in folds_to_run:
                error = self._run_cv_fold(m, fold_number, cv_prediction)
                error_metric = error_metric.append({'model': m, 'fold': fold_number, 'error': error}, ignore_index=True)
            total_error = round(self.metric(cv_prediction, self.train.ix[:, self.y_column]), 6)
            self.write_csv(cv_prediction, re.sub('.csv', '', filename) + model_key + str(total_error) + '_cv.csv')
        return error_metric

    def _run_final(self, models, filename, model_key):
        for m in models:
            m.fit(X=self.train.ix[:, self.X_columns], y=self.train.ix[:, self.y_column])
            prediction = m.predict(X=self.test.ix[:, self.X_columns])
            self.write_csv(prediction, re.sub('.csv', '', filename) + model_key + '_full.csv')

    # def _log_results(self):

    def _assign_cv_fold(self):
        if self.strategy['type'] == 'shuffle':
            records_per_fold = np.ceil(self.train.shape[0] / self.strategy['fold_cnt'])
            fold_numbers = np.random.permutation(range(self.strategy['fold_cnt'])*records_per_fold)
            self.train['fold'] = fold_numbers[:self.train.shape[0]]
            self.cv_index_col = 'fold'
        else:
            raise ValueError("Input strategy['type'] is not yet implemented!")

    def _run_cv_fold(self, m, fold_number, cv_prediction):
        train_filter = self.train['fold'] == fold_number
        cv_filter = [not x for x in train_filter]
        m.fit(X=self.train.ix[train_filter, self.X_columns], y=self.train.ix[train_filter, self.y_column])
        prediction = m.predict(X=self.train.ix[cv_filter, self.X_columns])
        cv_prediction[cv_filter] = prediction
        return self.metric(prediction, self.train.ix[cv_filter, self.y_column])

    @staticmethod
    def write_csv(data, _file):
        with open(_file, 'wb') as f:
            writer = csv.writer(f)
            writer.writerows(data)