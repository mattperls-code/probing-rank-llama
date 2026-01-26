import zarr
from zarr.storage import LocalStore
import json
import random
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import os
import matplotlib.pyplot as plt

data_path = "../../data/output-3k"

output_path = "../results"

feature_names: set[str] = [
    # "covered_query_term_count",         # number of query terms that occur in the doc
    "covered_query_term_ratio",         # ratio of query terms that occur in the doc
    # "doc_len",                          # number of terms in the document
    "log_doc_len",                      # log of number of terms in the document
    # "inverse_doc_len",                  # inverse of number of terms in the document
    "inverse_log_doc_len",              # inverse of log of number of terms in the document
    # "query_len",                        # number of terms in the query
    # "sum_term_count",                   # sum of number of times a query term appears in the doc
    # "avg_term_count",                   # avg of number of times a query term appears in the doc
    # "sum_tf",                           # sum of probability a query term appears in the doc
    "avg_tf",                           # avg of probability a query term appears in the doc
    # "sum_idf",                          # sum of log of inverse document frequency of a query term if it appears in the doc
    "avg_idf",                          # avg of log of inverse document frequency of a query term if it appears in the doc
    "covered_idf_ratio",                # sum of idf of terms in doc and query divided by sum of idf of terms in query
    # "sum_tfidf",                        # sum of product of tf and idf of a query term
    "avg_tfidf",                        # avg of product of tf and idf of a query term
    "bm25",                             # bm25 score
    # "log_bm25",                         # log of bm25 score
    # "sqrt_bm25",                        # sqrt of bm25 score
    "direchlet_ql",                     # query likelihood score with direchlet smoothing
    "jelinek_mercer_ql",                # query likelihood score with jelinek-mercer smoothing
    "kl_wrt_query",                     # kl divergence using query for true distribution and smoothed doc for approx
    "kl_wrt_doc",                       # kl divergence using doc for true distribution and smoothed query for approx
    "random_noise_10",                  # a random sample on -10 to 10 to detect overfitting
    "random_noise_50",                  # a random sample on -50 to 50 to detect overfitting
]

ridge_alpha = 10000.0

training_data_split = 2500 # samples used to compute the regression
# validation_data_split = total data - training data, these samples are used to compute r^2 of the regression

def main():
    activation_store = LocalStore(f"{data_path}/activations.zarr")
    activation_data = zarr.open(activation_store, mode="r")

    with open(f"{data_path}/features.json", "r") as features_file: feature_data = json.load(features_file)

    print("Retrieved activation and feature data")

    # shuffle pairs since they are ordered by all rel then all nrel

    random.seed(3.14159)
    perm = np.random.permutation(len(feature_data))

    shuffled_activation_data = zarr.zeros_like(activation_data)
    for new_index, old_index in enumerate(perm): shuffled_activation_data[new_index] = activation_data[old_index]
    activation_data = shuffled_activation_data

    feature_data = [ feature_data[index] for index in perm ]

    num_pairs, num_layers, num_neurons = activation_data.shape

    regression_xs = activation_data[:]
    regression_y = np.array([
        [feature_data[pair_index][feature_name] for feature_name in feature_names]
        for pair_index in range(num_pairs)
    ], dtype=np.float16)

    print("Formatted data for ridge regression")

    feature_r2_maps = []

    scaler = StandardScaler()

    for layer_index in range(num_layers):
        regression_x = scaler.fit_transform(regression_xs[:, layer_index, :])

        model = Ridge(alpha=ridge_alpha)
        model.fit(regression_x[:training_data_split, :], regression_y[:training_data_split, :])

        y_prediction = model.predict(regression_x)

        feature_r2_maps.append({
            feature_names[feature_index]: r2_score(regression_y[training_data_split:, feature_index], y_prediction[training_data_split:, feature_index])
            for feature_index in range(len(feature_names))
        })

    feature_r2_progressions = { feature_name: [] for feature_name in feature_names }

    for feature_r2_map in feature_r2_maps:
        for feature_name, r2 in feature_r2_map.items(): feature_r2_progressions[feature_name].append(r2)

    print("Calculated r-squared progressions")

    os.makedirs(output_path, exist_ok=True)

    for feature_name, r2_values in feature_r2_progressions.items():
        plt.clf()
        plt.plot(range(1, num_layers + 1, 1), r2_values)
        plt.xlabel("Layer Index")
        plt.xlim(1, num_layers)
        plt.ylabel("R Squared")
        plt.ylim(0.0, 1.0)
        plt.title(feature_name)
        plt.grid(True)
        plt.savefig(f"{output_path}/{feature_name}.png")
        plt.close()

    plt.clf()

    plt.figure(figsize=(16, 10))

    for feature_index, (feature_name, r2_values) in enumerate(feature_r2_progressions.items(), start=1):
        plt.plot(
            range(1, num_layers + 1, 1),
            r2_values,
            marker="o",
            linewidth=0.5,
            label=f"{feature_index}: {feature_name}"
        )

        plt.text(
            num_layers + 0.5,
            r2_values[-1],
            str(feature_index),
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center"
        )

    plt.xlabel("Layer Index")
    plt.xlim(1, num_layers + 1)
    plt.ylabel("R Squared")
    plt.ylim(0.0, 1.0)
    plt.title("Several Classical IR Features")
    plt.grid(True)

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )
    plt.tight_layout()

    plt.savefig(f"{output_path}/all.png")
    plt.close()

if __name__ == "__main__": main()