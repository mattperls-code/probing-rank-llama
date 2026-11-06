import zarr
from zarr.storage import LocalStore
import json
import random
import numpy as np

from util import feature_names, stratified_permutation, stratified_bootstrap, plot_r2_progressions, calculate_r2_progressions

data_path = "../../data/output-3k"
output_path = "../main-results"

training_data_split = 2500 # samples used to compute the regression
# validation_data_split = total data - training data, these samples are used to compute r^2 of the regression

def boostrap_training_set(repetitions: int = 1000):
    activation_store = LocalStore(f"{data_path}/activations.zarr")
    activation_data = zarr.open(activation_store, mode="r")

    with open(f"{data_path}/features.json", "r") as features_file: feature_data = json.load(features_file)

    print("Retrieved activation and feature data")

    # shuffle pairs since they are ordered by all rel then all nrel

    np.random.seed(314159)
    perm = stratified_permutation(feature_data, training_data_split)

    feature_data = [ feature_data[index] for index in perm ]

    num_pairs, num_layers, num_neurons = activation_data.shape

    # determine target types before converting to np
    feature_is_binary = {
        feature_name: isinstance(feature_data[0][feature_name], bool)
        for feature_name in feature_names
    }

    y = np.array([
        [ feature_data[pair_index][feature_name] for feature_name in feature_names ]
        for pair_index in range(num_pairs)
    ], dtype=np.float16)

    training_y = y[:training_data_split, :]
    validation_y = y[training_data_split:, :]

    print("Formatted data for regression")

    for _ in range(repetitions):
        bootstrap_training_indices = stratified_bootstrap(feature_data[:training_data_split])

        bootstrap_training_xs = activation_data[perm[:training_data_split][bootstrap_training_indices], :, :]

        print(bootstrap_training_xs)
        print(bootstrap_training_xs.shape)

        raise ValueError()

if __name__ == "__main__":
    boostrap_training_set()