import time
import json
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from typing import List


def find_neighbors_within_radius(distances, train_labels, radius):
    """
    Find neighbors within a given radius.
    :param distances: Distance matrix
    :param train_labels: Training labels
    :param radius: Radius within which to find neighbors
    :return: List of neighbors within the radius for each instance
    """
    # Create a boolean mask where True indicates distances within the radius
    mask = distances <= radius
    # Use the mask to filter train_labels
    neighbors_list = [train_labels[mask[:, i]] for i in range(mask.shape[1])]
    return neighbors_list


def scale_features(X_train, X_val, X_test):
    """
    Scale features using StandardScaler.
    :param X_train: Training features
    :param X_val: Validation features
    :param X_test: Test features
    :return: Scaled training, validation, and test features
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    X_test = scaler.fit_transform(X_test)
    return X_train, X_val, X_test


def compute_distance_matrix(X1, X2):
    """
    Compute the Euclidean distance matrix.
    :param X1: First set of instances
    :param X2: Second set of instances
    :return: Distance matrix
    """
    return np.sqrt(np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2))


def get_radius_range(val_distance_matrix, num_points):
    """
    Generate a range of radii to test.
    :param val_distance_matrix: Validation distance matrix
    :param num_points: Number of points in the radius range
    :return: Range of radii
    """
    min_radius = np.min(val_distance_matrix)
    if min_radius == 0:
        min_radius += 0.00001
    max_radius = np.max(val_distance_matrix)
    return np.logspace(np.log10(min_radius), np.log10(max_radius), num_points)


def get_class_distribution(y):
    """
    Calculate the class distribution in the given labels.
    :param y: Labels
    :return: Class distribution as a dictionary
    """
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))


def predict_classes(neighbors_list, class_distribution):
    """
    Predict classes based on neighbors.
    :param neighbors_list: List of neighbors for each instance
    :param class_distribution: Class distribution to use for tie-breaking
    :return: Predicted classes
    """
    predictions = []
    for i, neighbors in enumerate(neighbors_list):
        if neighbors.size > 0:
            unique, counts = np.unique(neighbors, return_counts=True)
            max_count = np.max(counts)
            most_frequent_classes = unique[counts == max_count]
            if len(most_frequent_classes) > 1:
                # Choose based on class distribution in case of a tie
                weights = [class_distribution[c] for c in most_frequent_classes]
                predicted_class = np.random.choice(most_frequent_classes, p=np.array(weights) / np.sum(weights))
            else:
                predicted_class = most_frequent_classes[0]
        else:
            # Choose based on class distribution if no neighbors are found
            predicted_class = np.random.choice(list(class_distribution.keys()),
                                               p=np.array(list(class_distribution.values())) / np.sum(
                                                   list(class_distribution.values())))
        predictions.append(predicted_class)
    return predictions


def find_best_radius(val_distance_matrix, y_train, y_val, class_distribution):
    """
    Find the best radius for the classifier using validation data.
    :param val_distance_matrix: Validation distance matrix
    :param y_train: Training labels
    :param y_val: Validation labels
    :param patience: Number of iterations with no improvement to stop early
    :param class_distribution: Class distribution for tie-breaking
    :return: Best radius
    """
    best_radius = None
    best_accuracy = 0
    sqrt_train_size = int(np.sqrt(val_distance_matrix.shape[0]))
    radius_range = get_radius_range(val_distance_matrix, sqrt_train_size)

    patience = int(np.sqrt(sqrt_train_size))
    previous = 0
    no_improvement = 0

    for radius in radius_range:
        val_neighbors_list = find_neighbors_within_radius(val_distance_matrix, y_train, radius)
        y_val_pred = predict_classes(val_neighbors_list, class_distribution)
        accuracy = accuracy_score(y_val, y_val_pred)
        print(f'Radius: {radius}, Validation Accuracy: {accuracy}')
        # Update best accuracy and radius if current accuracy is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_radius = radius

        # Check for early stopping
        if previous < accuracy:
            no_improvement = 0
        elif previous != accuracy:
            no_improvement += 1
        previous = accuracy

        if no_improvement >= patience:
            break
    print(best_radius)
    return best_radius


def classify_with_NNR(data_trn: str, data_vld: str, df_tst: DataFrame) -> List:
    """
    Classify data using the Nearest Neighbor Radius (NNR) method.
    :param data_trn: Path to training data
    :param data_vld: Path to validation data
    :param df_tst: Test data DataFrame
    :return: Predictions for the test data
    """
    print(f'Starting classification with {data_trn}, {data_vld}, predicting on {len(df_tst)} instances')

    # Load training and validation datasets
    df_train = pd.read_csv(data_trn)
    df_val = pd.read_csv(data_vld)

    X_train, y_train = df_train.drop(['class'], axis=1).values, df_train['class'].values
    X_val, y_val = df_val.drop(['class'], axis=1).values, df_val['class'].values
    X_test = df_tst.values

    # Scale the features
    X_train, X_val, X_test = scale_features(X_train, X_val, X_test)

    # Compute distance matrices for validation and test sets
    val_distance_matrix = compute_distance_matrix(X_train, X_val)
    test_distance_matrix = compute_distance_matrix(X_train, X_test)

    # Calculate class distribution in the training data
    train_class_distribution = get_class_distribution(y_train)

    # Find the best radius using the validation set
    best_radius = find_best_radius(val_distance_matrix, y_train, y_val, class_distribution=train_class_distribution)

    # Use the best radius to classify test data
    test_neighbors_list = find_neighbors_within_radius(test_distance_matrix, y_train, best_radius)
    predictions = predict_classes(test_neighbors_list, train_class_distribution)

    return predictions


students = {'id1': '207388331', 'id2': '208277343'}

if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    df = pd.read_csv(config['data_file_test'])
    predicted = classify_with_NNR(config['data_file_train'], config['data_file_validation'], df.drop(['class'], axis=1))

    labels = df['class'].values
    if not predicted:  # Empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert len(labels) == len(predicted)  # Make sure you predict label for all test instances
    print(f'Test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'Total time: {round(time.time() - start, 0)} sec')
