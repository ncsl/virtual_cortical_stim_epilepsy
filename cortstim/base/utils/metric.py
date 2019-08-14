import numpy as np

from sklearn.metrics.classification import _check_targets, \
    check_consistent_length, count_nonzero, _weighted_sum


def degree_of_agreement(y_true, y_pred, normalize=True, sample_weight=None):
    # Compute accuracy for each possible representation
    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    check_consistent_length(y_true, y_pred, sample_weight)

    if y_type.startswith('multilabel'):
        with np.errstate(divide='ignore', invalid='ignore'):
            # oddly, we may get an "invalid" rather than a "divide" error here
            pred_and_true = count_nonzero(
                y_true.multiply(y_pred), axis=1)  # cez inters eez
            len_true = count_nonzero(y_true)

            not_ytrue = 1 - y_true
            pred_and_nottrue = count_nonzero(not_ytrue.multiply(y_pred), axis=1)
            len_nottrue = count_nonzero(not_ytrue)

            pred_or_true = count_nonzero(y_true + y_pred, axis=1)

            # compute the doa statistic
            score = pred_and_true / len_true - pred_and_nottrue / len_nottrue
            # score = pred_and_true / pred_or_true
            print(score)
            score[pred_or_true == 0.0] = 1.0
            print(score)
    else:
        # oddly, we may get an "invalid" rather than a "divide" error here
        pred_and_true = np.count_nonzero(
            y_true * y_pred, axis=0)  # cez inters eez
        len_true = np.count_nonzero(y_true)

        not_ytrue = np.subtract(1, y_true)
        pred_and_nottrue = np.count_nonzero(not_ytrue * y_pred, axis=0)
        len_nottrue = np.count_nonzero(not_ytrue)

        # compute the doa statistic
        score = pred_and_true / len_true - pred_and_nottrue / len_nottrue

    return _weighted_sum(score, sample_weight, normalize)
