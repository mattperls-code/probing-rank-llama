import zarr
from zarr.storage import LocalStore
import json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score

from util import ridge_alpha, stratified_permutation, stratified_bootstrap, calculate_r2_progressions, plot_r2_progressions_with_ci

data_path = "../../data/output-3k"
output_path = "../bootstrap-ci-results"

# 1k repetitions x 32 layers x 4k neurons is expensive, so we only do bootstrap ci for a few target features
feature_names = [ "score", "bm25", "covered_idf_ratio" ]

training_data_split = 2500


def bootstrap_validation_set(repetitions: int = 10):
    activation_store = LocalStore(f"{data_path}/activations.zarr")
    activation_data = zarr.open(activation_store, mode="r")

    num_pairs, num_layers, num_neurons = activation_data.shape

    with open(f"{data_path}/features.json", "r") as features_file: feature_data = json.load(features_file)

    print("Retrieved activation and feature data")

    # shuffle pairs since they are ordered by all rel then all nrel

    np.random.seed(314159)
    perm = stratified_permutation(feature_data, training_data_split)

    training_xs = activation_data[perm[:training_data_split], :, :]
    validation_xs = activation_data[perm[training_data_split:], :, :]

    feature_data = [ feature_data[index] for index in perm ]

    y = np.array([
        [ feature_data[pair_index][feature_name] for feature_name in feature_names ]
        for pair_index in range(num_pairs)
    ], dtype=np.float16)

    training_y = y[:training_data_split, :]
    validation_y = y[training_data_split:, :]

    print("Formatted data for regression")

    # [feature][layer] = scaler, ridge model
    regressions = {}

    for feature_index, feature_name in enumerate(feature_names):
        regressions[feature_name] = []

        for layer_index in range(num_layers):
            scaler = StandardScaler()

            training_x = scaler.fit_transform(training_xs[:, layer_index, :])

            model = Ridge(alpha=ridge_alpha, solver="auto")
            model.fit(training_x, training_y[:, feature_index])

            regressions[feature_name].append((scaler, model)) # redundant to have same scaler for every layer, but pretty cheap anyway

    print("Finished fitting model")

    r2_progressions_samples = []

    for i in range(repetitions):
        print(f"Bootstrap Repetition {i+1}/{repetitions}")

        bootstrap_validation_indices = stratified_bootstrap(feature_data[training_data_split:])

        bootstrap_validation_xs = validation_xs[bootstrap_validation_indices, :, :]
        bootstrap_validation_y = validation_y[bootstrap_validation_indices, :]

        r2_progressions = {}

        for feature_index, feature_name in enumerate(feature_names):
            r2_progressions[feature_name] = []

            for layer_index in range(num_layers):
                scaler, model = regressions[feature_name][layer_index]

                validation_predictions = model.predict(scaler.transform(bootstrap_validation_xs[:, layer_index, :]))

                r2_progressions[feature_name].append(r2_score(bootstrap_validation_y[:, feature_index], validation_predictions))

        r2_progressions_samples.append(r2_progressions)

    plot_r2_progressions_with_ci(output_path, "bootstrap-validation-data", r2_progressions_samples)


def boostrap_training_set(repetitions: int = 1000):
    activation_store = LocalStore(f"{data_path}/activations.zarr")
    activation_data = zarr.open(activation_store, mode="r")

    num_pairs, num_layers, num_neurons = activation_data.shape

    with open(f"{data_path}/features.json", "r") as features_file: feature_data = json.load(features_file)

    print("Retrieved activation and feature data")

    # shuffle pairs since they are ordered by all rel then all nrel

    np.random.seed(314159)
    perm = stratified_permutation(feature_data, training_data_split)

    validation_xs = activation_data[perm[training_data_split:], :, :]

    # used for util, in this case all regressions are over continuous targets
    feature_is_binary = {
        feature_name: False
        for feature_name in feature_names
    }

    feature_data = [ feature_data[index] for index in perm ]

    y = np.array([
        [ feature_data[pair_index][feature_name] for feature_name in feature_names ]
        for pair_index in range(num_pairs)
    ], dtype=np.float16)

    validation_y = y[training_data_split:, :]

    print("Formatted data for regression")

    r2_progressions_samples = []

    for i in range(repetitions):
        print(f"Bootstrap Repetition {i+1}/{repetitions}")

        bootstrap_training_indices = stratified_bootstrap(feature_data[:training_data_split])

        bootstrap_training_xs = activation_data[perm[:training_data_split][bootstrap_training_indices], :, :]
        bootstrap_training_y = y[bootstrap_training_indices, :]

        _, validation_r2_progressions = calculate_r2_progressions(
            num_layers,
            feature_is_binary,
            bootstrap_training_xs,
            validation_xs,
            bootstrap_training_y,
            validation_y,
            feature_names=feature_names
        )

        r2_progressions_samples.append(validation_r2_progressions)

    plot_r2_progressions_with_ci(output_path, "bootstrap-training-data", r2_progressions_samples)

if __name__ == "__main__":
    bootstrap_validation_set(200)
    boostrap_training_set(200)