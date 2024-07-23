import time
import json
import numpy as np
from utils import evaluate_clustering_result
import pickle

# Hard-coded minimum similarity threshold
min_similarity = 0.738

# Function to load a dataset using pickle
def load_from_pickle(path):
    with open(path, 'rb') as fin:
        dataset = pickle.load(fin)
        return dataset

# Function to calculate cosine similarity between two vectors
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


# Function to update the centroid of a cluster
def update_centroid(cluster, elements):
    return np.mean([elements[key] for key in cluster], axis=0)


# Function to perform clustering on the dataset
def cluster_data(features_file, min_cluster_size, max_iterations):
    print(f'Starting clustering images in {features_file}')

    # Load the dataset from a pickle file
    dataset = load_from_pickle(features_file)
    # Convert dataset values to numpy arrays for easier manipulation
    elements = {key: np.array(value) for key, value in dataset.items()}

    clusters = {}  # Dictionary to hold clusters
    cluster_centroids = {}  # Dictionary to hold centroids of clusters
    element_cluster_map = {}  # Dictionary to map elements to their respective clusters

    for iteration in range(max_iterations):
        changes = False  # Flag to check if any changes occur in an iteration

        for key, element in elements.items():
            max_similarity = -1
            best_cluster = None

            # Find the most similar cluster for the current element
            for cluster_id, centroid in cluster_centroids.items():
                similarity = cosine_similarity(element, centroid)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_cluster = cluster_id

            # If the most similar cluster is above the similarity threshold
            if best_cluster is not None and max_similarity > min_similarity:
                current_cluster = element_cluster_map.get(key)
                if current_cluster != best_cluster:
                    # Remove the element from its current cluster if it has one
                    if current_cluster is not None:
                        clusters[current_cluster].remove(key)
                        if len(clusters[current_cluster]) > 0:
                            cluster_centroids[current_cluster] = update_centroid(clusters[current_cluster], elements)
                        else:
                            del clusters[current_cluster]
                            del cluster_centroids[current_cluster]

                    # Add the element to the best matching cluster
                    if best_cluster not in clusters:
                        clusters[best_cluster] = []

                    clusters[best_cluster].append(key)
                    cluster_centroids[best_cluster] = update_centroid(clusters[best_cluster], elements)
                    element_cluster_map[key] = best_cluster
                    changes = True
            else:
                # Create a new cluster for the element if it doesn't fit well into any existing cluster
                new_cluster_id = len(clusters) + 1
                clusters[new_cluster_id] = [key]
                cluster_centroids[new_cluster_id] = element
                element_cluster_map[key] = new_cluster_id
                changes = True

        # Stop the iteration if no changes were made
        if not changes:
            break

    # Filter out clusters that are smaller than the minimum cluster size
    final_clusters = {cid: members for cid, members in clusters.items() if len(members) >= min_cluster_size}

    return final_clusters


if __name__ == '__main__':
    start = time.time()

    # Load configuration from JSON file
    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    # Perform clustering on the dataset
    result = cluster_data(config['features_file'],
                          config['min_cluster_size'],
                          config['max_iterations'])

    # Evaluate the clustering result
    evaluation_scores = evaluate_clustering_result(config['labels_file'], result)
    print(f'Total time: {round(time.time() - start, 0)} sec')
