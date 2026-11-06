import zarr
from zarr.storage import LocalStore
import json
import random
import numpy as np

from util import feature_names, plot_r2_progressions, calculate_r2_progressions

data_path = "../../data/output-3k"
output_path = "../cross-class-results"

if __name__ == "__main__":
    activation_store = LocalStore(f"{data_path}/activations.zarr")
    activation_data = zarr.open(activation_store, mode="r")

    num_pairs, num_layers, num_neurons = activation_data.shape

    with open(f"{data_path}/features.json", "r") as features_file: feature_data = json.load(features_file)

    print("Retrieved activation and feature data")

    rel_xs = activation_data[:500, :, :]
    nrel_xs = activation_data[500:, :, :]

    # determine target types before converting to np
    feature_is_binary = {
        feature_name: isinstance(feature_data[0][feature_name], bool)
        for feature_name in feature_names
    }

    y = np.array([
        [ feature_data[pair_index][feature_name] for feature_name in feature_names ]
        for pair_index in range(num_pairs)
    ], dtype=np.float16)

    rel_y = y[:500, :]
    nrel_y = y[500:, :]

    print("Formatted data for regression")

    # the probe is trained on relevant pairs to predict non-relevant pairs
    _, rel_predicting_nrel_r2_progressions = calculate_r2_progressions(
        num_layers,
        feature_is_binary,
        rel_xs,
        nrel_xs,
        rel_y,
        nrel_y
    )

    # the probe is trained on non-relevant pairs to predict relevant pairs
    _, nrel_predicting_rel_r2_progressions = calculate_r2_progressions(
        num_layers,
        feature_is_binary,
        nrel_xs,
        rel_xs,
        nrel_y,
        rel_y
    )

    print("Calculated r-squared progressions")

    plot_r2_progressions(output_path, "rel_predicting_nrel", rel_predicting_nrel_r2_progressions)
    plot_r2_progressions(output_path, "nrel_predicting_rel", nrel_predicting_rel_r2_progressions)