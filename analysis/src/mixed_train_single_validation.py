import zarr
from zarr.storage import LocalStore
import json
import random
import numpy as np

from util import feature_names, stratified_permutation, plot_r2_progressions, calculate_r2_progressions

data_path = "../../data/output-3k"
output_path = "../mixed-train-single-validation-results"

training_data_split = 2500 # samples used to compute the regression
# validation_data_split = total data - training data, these samples are used to compute r^2 of the regression

if __name__ == "__main__":
    activation_store = LocalStore(f"{data_path}/activations.zarr")
    activation_data = zarr.open(activation_store, mode="r")

    num_pairs, num_layers, num_neurons = activation_data.shape

    with open(f"{data_path}/features.json", "r") as features_file: feature_data = json.load(features_file)

    # discard binary predictions for this, breaks auc
    feature_names.remove("relevant")
    feature_names.remove("random_class")
    feature_names.remove("weighted_random_class")

    print("Retrieved activation and feature data")

    # shuffle pairs since they are ordered by all rel then all nrel

    np.random.seed(314159)
    perm = stratified_permutation(feature_data, training_data_split)

    activation_data = activation_data[perm, :, :]

    feature_data = [ feature_data[index] for index in perm ]

    training_xs = activation_data[:training_data_split, :, :]
    validation_xs = activation_data[training_data_split:, :, :]

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

    validation_rel_mask = np.array([
        features["relevant"]
        for features in feature_data[training_data_split:]
    ], dtype=bool)

    validation_nrel_mask = ~validation_rel_mask

    rel_validation_xs = validation_xs[validation_rel_mask, :, :]
    rel_validation_y = validation_y[validation_rel_mask, :]

    nrel_validation_xs = validation_xs[validation_nrel_mask, :, :]
    nrel_validation_y = validation_y[validation_nrel_mask, :]

    print("Formatted data for regression")

    _, rel_validation_r2_progressions = calculate_r2_progressions(
        num_layers,
        feature_is_binary,
        training_xs,
        rel_validation_xs,
        training_y,
        rel_validation_y
    )

    _, nrel_validation_r2_progressions = calculate_r2_progressions(
        num_layers,
        feature_is_binary,
        training_xs,
        nrel_validation_xs,
        training_y,
        nrel_validation_y
    )

    print("Calculated r-squared progressions")

    plot_r2_progressions(output_path, "rel_validation", rel_validation_r2_progressions)
    plot_r2_progressions(output_path, "nrel_validation", nrel_validation_r2_progressions)