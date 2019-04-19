"""The code was copied from liorshk's 'face_pytorch' repository:
    https://github.com/liorshk/facenet_pytorch/blob/master/eval_metrics.py

    Which in turn was copied from David Sandberg's 'facenet' repository:
        https://github.com/davidsandberg/facenet/blob/master/src/lfw.py#L34
        https://github.com/davidsandberg/facenet/blob/master/src/facenet.py#L424
"""

import numpy as np
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from scipy import interpolate


def evaluate_lfw(distances, labels, num_folds=10):
    """Evaluates on the Labeled Faces in the Wild dataset using KFold cross validation based on the Euclidean
    distance as a metric.

    Note: "Val@FAL=0.001" means the rate that faces are successfully accepted (TP/(TP+FN)) when the rate that faces are
     incorrectly accepted (FP/(TN+FP)) is 0.001.
        https://github.com/davidsandberg/facenet/issues/288#issuecomment-305961018

    Args:
        distances: numpy array of the pairwise distances calculated from the LFW pairs.
        labels: numpy array containing the correct result of the LFW pairs belonging to the same identity or not.
        num_folds (int): Number of folds for KFold cross-validation, defaults to 10 folds.

    Returns:
        true_positive_rate: Mean value of all true positive rates across all cross validation folds.
        false_positive_rate: Mean value of all false positive rates across all cross validation folds.
        accuracy: Array of accuracies per each fold in cross validation.
        roc_auc: Area Under the Curve metric resulting from the KFold cross validation.
        best_distance_threshold: The Euclidean distance value that had the best performing accuracy on the lfw dataset.
        val: Accuracy when far is set to a specific probability value.
        val_std: Standard deviation of val.
        far: False Accept Rate, rate of face pairs that are different and yet have a distance that is below a certain
             threshold so they would be predicted to be faces of the same person. Default far = 0.001
    """
    thresholds_roc = np.arange(0, 30, 0.01)
    true_positive_rate, false_positive_rate, accuracy, best_distance_threshold = calculate_roc_values(
        thresholds=thresholds_roc, distances=distances, labels=labels, num_folds=num_folds
    )
    roc_auc = auc(false_positive_rate, true_positive_rate)

    thresholds_val = np.arange(0, 30, 0.001)
    val, val_std, far = calculate_val(
        thresholds_val=thresholds_val, distances=distances, labels=labels, far_target=1e-3, num_folds=num_folds
    )

    return true_positive_rate, false_positive_rate, accuracy, roc_auc, best_distance_threshold, val, val_std, far


def calculate_roc_values(thresholds, distances, labels, num_folds=10):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    true_positive_rates = np.zeros((num_folds, num_thresholds))
    false_positive_rates = np.zeros((num_folds, num_thresholds))
    accuracy = np.zeros(num_folds)

    indices = np.arange(num_pairs)

    for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best distance threshold for the k-fold cross validation using the train set
        accuracies_trainset = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds):
            _, _, accuracies_trainset[threshold_index] = calculate_accuracy(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        best_threshold_index = np.argmax(accuracies_trainset)
        # Test on test set using the best distance threshold
        for threshold_index, threshold in enumerate(thresholds):
            true_positive_rates[fold_index, threshold_index], false_positive_rates[fold_index, threshold_index], _ = \
                calculate_accuracy(threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set])

        _, _, accuracy[fold_index] = calculate_accuracy(
            threshold=thresholds[best_threshold_index], dist=distances[test_set], actual_issame=labels[test_set]
        )

        true_positive_rate = np.mean(true_positive_rates, 0)
        false_positive_rate = np.mean(false_positive_rates, 0)
        best_distance_threshold = thresholds[best_threshold_index]

    return true_positive_rate, false_positive_rate, accuracy, best_distance_threshold


def calculate_accuracy(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)

    true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
    false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    true_positive_rate = 0 if (true_positives + false_negatives == 0) else \
        float(true_positives) / float(true_positives + false_negatives)

    false_positive_rate = 0 if (false_positives + true_negatives == 0) else \
        float(false_positives) / float(false_positives + true_negatives)

    accuracy = float(true_positives + true_negatives) / dist.size

    return true_positive_rate, false_positive_rate, accuracy


def calculate_val(thresholds_val, distances, labels, far_target=1e-3, num_folds=10):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds_val)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    val = np.zeros(num_folds)
    far = np.zeros(num_folds)

    indices = np.arange(num_pairs)

    for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the euclidean distance threshold that gives false acceptance rate (far) = far_target
        far_train = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds_val):
            _, far_train[threshold_index] = calculate_val_far(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds_val, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_index], far[fold_index] = calculate_val_far(
            threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
        )

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)

    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    num_same = np.sum(actual_issame)
    num_diff = np.sum(np.logical_not(actual_issame))

    if num_diff == 0:
        num_diff = 1
    if num_same == 0:
        return 0, 0

    val = float(true_accept) / float(num_same)
    far = float(false_accept) / float(num_diff)

    return val, far
