import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.model_selection import KFold


def evaluate_lfw(distances, labels, num_folds=10):
    """Evaluates on the Labeled Faces in the Wild dataset using KFold cross validation based on the Euclidean
    distance as a metric.

    Args:
        distances: numpy array of the pairwise distances calculated from the LFW pairs.
        labels: numpy array containing the correct result of the LFW pairs belonging to the same identity or not.
        num_folds (int): Number of folds for KFold cross-validation, defaults to 10 folds.

    Returns:
        true_positive_rate: True Positive Rate metric resulting for the KFold cross validation.
        false_positive_rate: False Positive Rate metric resulting for the KFold cross validation.
        accuracy: Accuracy metric resulting for the KFold cross validation.
    """
    thresholds_roc = np.arange(0, 30, 0.01)
    true_positive_rate, false_positive_rate, accuracy = calculate_roc_values(
        thresholds=thresholds_roc, distances=distances, labels=labels, num_folds=num_folds
    )

    return true_positive_rate, false_positive_rate, accuracy


def calculate_roc_values(thresholds, distances, labels, num_folds=10):
    """Calculates the True Positive Rate (TPR) and False Positive Rate (FAR) metrics for use in the Receiver Operating
    Characteristic (ROC) curve based on the best performing Euclidean distance threshold.

    Args:
        thresholds: numpy array containing the list of Euclidean distance thresholds for use in KFold cross validation.
        distances: numpy array of the pairwise distances calculated from the LFW pairs.
        labels: numpy array containing the correct result of the LFW pairs belonging to the same identity or not.
        num_folds (int): Number of folds for KFold cross-validation, defaults to 10 folds.

    Returns:
        true_positive_rate: True Positive Rate metric resulting from best performing Euclidean distance threshold.
        false_positive_rate: False Positive Rate metric resulting from best performing Euclidean distance threshold.
        accuracy: Accuracy metric resulting from best performing Euclidean distance threshold.

    """
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

    return true_positive_rate, false_positive_rate, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    """Calculates the True Positive Rate (TPR) and False Positive Rate (FAR) metrics for each Euclidean distance
    threshold as part of the KFold cross validation process.

    Args:
        threshold: Euclidean distance value to be used as a metric for cross validation.
        dist: numpy array of the pairwise distances calculated from the LFW pairs from either the train set or test set
              of the cross validation process.
        actual_issame: The correct result of the LFW pairs belonging to the same identity or not.

    Returns:
        true_positive_rate: True Positive Rate metric resulting for the KFold cross validation.
        false_positive_rate: False Positive Rate metric resulting for the KFold cross validation.
        accuracy: Accuracy metric resulting for the KFold cross validation.
    """
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


def plot_roc_lfw(false_positive_rate, true_positive_rate, figure_name="roc.png"):
    """Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        false_positive_rate: False positive rate
        true_positive_rate:
        figure_name (str): Name of the image file of the resulting ROC curve plot.
    """
    roc_auc = auc(false_positive_rate, true_positive_rate)
    fig = plt.figure()
    plt.plot(
        false_positive_rate, true_positive_rate, color='red', lw=2, label="ROC Curve (area = {:.2f})".format(roc_auc)
    )
    plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)
