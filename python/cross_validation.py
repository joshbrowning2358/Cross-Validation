import pandas as pd
import numpy as np
import re

pd.options.mode.chained_assignment = None


class CrossValidation:
    """
    Class to perform automatic cross-validation when given data and a
    sampling approach.
    """

    def __init__(self, X_train, target, X_test, metric, cv_index=None, validation_index=None,
                 cv_time=False, id_col_name=None, logged=False):
        """
        :param metric: A function accepting two vectors and computing an error metric.
        """
        # if not isinstance(X_train, pd.DataFrame):
        #     raise TypeError('data must be of type DataFrame!')
        self.X_train = X_train
        self.X_test = X_test
        self.target = target
        self.metric = metric
        if cv_index is not None and validation_index is None:
            self.fold = cv_index
            self.cv_type = 'cross_validation'
        elif cv_index is None and validation_index is not None:
            self.fold = validation_index
            self.cv_type = 'single_validation'
        elif cv_index is not None and validation_index is not None:
            raise ValueError('Cannot provide both cv_indices and validation_indices!')
        else:
            self._assign_cv_fold()
            self.cv_type = 'cross_validation'
        if not isinstance(target, np.ndarray):
            raise TypeError('target must be a np.ndarray!')
        if cv_time:
            self.cv_type = 'time_validation'
        if id_col_name is not None:
            self.id_col_name = id_col_name
            self.id_col = X_test[id_col_name]
        else:
            self.id_col = range(self.X_test.shape[0])
            self.id_col_name = 'id'
        self.model_key = str(np.random.randint(100000)) + str(np.random.randint(100000))
        self.logged = logged

    def run(self, model, filename=None):
        """
        :param model: An instance with fit and predict methods.
        :param filename: Str, used to write out result files (from cv or train/test)
        """
        if filename is None:
            filename = "/data/models/" + self.model_key
        else:
            filename = re.sub("\\.csv", "_" + self.model_key, filename)
        if self.cv_type in ['time_validation', 'cross_validation']:
            score = self._run_cv(model, filename)
        elif self.cv_type == 'single_validation':
            score = self._run_validation(model, filename)
        else:
            raise TypeError('Validation type {} not yet implemented!'.format(self.cv_type))
        if self.logged:
            self._run_final(model, filename)

    def _run_cv(self, model, filename):
        error_metric = pd.DataFrame({'fold': [], 'error': []})
        folds_to_run = list(set(self.fold))
        cv_prediction = [np.nan] * self.X_train.shape[0]
        for fold_number in folds_to_run:
            cv_prediction, error = self._run_cv_fold(model, fold_number, cv_prediction)
            error_metric = error_metric.append({'fold': fold_number, 'error': error}, ignore_index=True)
            print 'Error for fold {} was: {}'.format(fold_number, str(error))
        total_error = round(self.metric(self.target, cv_prediction), 6)
        if self.logged:
            file_ = re.sub('.csv', '', filename) + '_' + self.model_key + str(total_error) + '_cv.csv'
            out = pd.concat([pd.DataFrame({self.id_col_name: self.id_col}), pd.DataFrame(cv_prediction)], axis=1)
            if out.shape[1] > 2: # categorical predictions
                out.columns = [self.id_col_name] + list(set(self.target))
            out.to_csv(file_, ',', header=True, index=False)
            print 'Cross validation results saved in ' + file_
        print 'Error was: ' + str(total_error)
        return error_metric

    def _run_validation(self, model, filename):
        fold_number = np.min(self.fold)
        cv_prediction = [np.nan] * self.X_train.shape[0]
        error = self._run_cv_fold(model, fold_number, cv_prediction)
        error_metric = pd.DataFrame({'fold': [fold_number], 'error': [error]})
        if self.logged:
            file_ = re.sub('.csv', '', filename) + '_' + self.model_key + str(round(error, 6)) + '_cv.csv'
            out = pd.concat([pd.DataFrame({self.id_col_name: self.id_col}), pd.DataFrame(cv_prediction)], axis=1)
            if out.shape[1] > 2: # categorical predictions
                out.columns = [self.id_col_name] + list(set(self.target))
            out.to_csv(file_, ',', header=True, index=False)
            print 'Validation results saved in ' + file_
        print 'Error was: ' + str(error)
        return error_metric

    def _run_final(self, model, filename):
        model.fit(X=self.X_train, y=self.target)
        prediction = model.predict(self.X_test)
        if self.logged:
            file_ = re.sub('.csv', '', filename) + '_' + self.model_key + '_full.csv'
            out = pd.concat([pd.DataFrame({self.id_col_name: self.id_col}), pd.DataFrame(prediction)], axis=1)
            if out.shape[1] > 2: # categorical predictions
                out.columns = [self.id_col_name] + list(set(self.target))
            out.to_csv(file_, ',', header=True, index=False)
            print 'Final predictions saved in ' + file_

    def _assign_cv_fold(self):
        fold_cnt = 4
        records_per_fold = self.X_train.shape[0] / fold_cnt + 1
        self.fold = np.random.permutation(range(fold_cnt)*records_per_fold)
        self.fold = self.fold[:self.X_train.shape[0]]

    def _run_cv_fold(self, m, fold_number, cv_prediction):
        cv_filter = self.fold == fold_number
        cv_indices = [i for i, x in enumerate(cv_filter) if x]
        train_filter = [not x for x in cv_filter]
        train_indices = [i for i, x in enumerate(train_filter) if x]
        m.fit(X=self.X_train[train_indices, :], y=self.target[train_indices])
        prediction = m.predict(X=self.X_train)
        cv_prediction = [pred if cv else old for pred, old, cv in zip(prediction, cv_prediction, cv_filter)]
        return cv_prediction,\
               self.metric(self.target[cv_indices], [pred for cond, pred in zip(cv_filter, cv_prediction) if cond])

    @staticmethod
    def write_csv(data, _file):
        data.to_csv(_file, ',', header=True, engine='python')
