import numpy as np

def multi_log_loss(actual, predicted):
    """
    Columns of predicted are assumed to be in alphabetical order of the class
    :param actual: List of observed classes
    :param predicted: Matrix with predicted probabilities
    :return: Score value (numeric)
    """
    if (not isinstance(actual, list)) and (not isinstance(actual, np.ndarray)):
        raise TypeError('actual must be a list!')
    if not isinstance(predicted, list):
        raise TypeError('predicted must be a list!')

    # Normalize predicted
    predicted = map(lambda x: x / sum(x), predicted)

    unique_values = list(set(actual))
    unique_values.sort()
    if len(unique_values) != len(predicted[0]):
        raise TypeError('Length of unique values of actual must match number of columns of predicted!')
    column = [i for row in actual for i, val in enumerate(unique_values) if row == val]
    predicted_log_probs = [np.log(x[i]) for x, i in zip(predicted, column)]
    return -np.mean(predicted_log_probs)


def rmse(predicted, actual):
    """
    Columns of predicted are assumed to be in alphabetical order of the class
    :param actual: List or numpy array of observed values
    :param predicted: List or numpy array with predicted probabilities
    :return: Score value (numeric)
    """
    if isinstance(actual, list):
        actual = np.array(actual)
    if isinstance(predicted, list):
        predicted = np.array(predicted)
    if not isinstance(actual, np.ndarray):
        raise TypeError('actual must be a list or numpy array!')
    if not isinstance(predicted, np.ndarray):
        raise TypeError('predicted must be a list or numpy array!')
    error = actual - predicted
    return np.sqrt(np.mean([e**2 for e in error]))