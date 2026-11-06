import os

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, roc_auc_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

feature_names = [
    "score",                            # the final pointwise ranking score produced by the model

    "relevant",                         # ground-truth relevance classification
    "random_class",                     # a random binary label picked 50/50 to detect overfitting
    "weighted_random_class",            # a random binary label picked from the real distribution of relevant pairs to detect overfitting

    # "covered_query_term_count",         # number of query terms that occur in the doc
    "covered_query_term_ratio",         # ratio of query terms that occur in the doc
    "doc_len",                          # number of terms in the document
    "log_doc_len",                      # log of number of terms in the document
    # "inverse_doc_len",                  # inverse of number of terms in the document
    # "inverse_log_doc_len",              # inverse of log of number of terms in the document
    # "query_len",                        # number of terms in the query
    # "sum_term_count",                   # sum of number of times a query term appears in the doc
    # "avg_term_count",                   # avg of number of times a query term appears in the doc
    "sum_tf",                           # sum of probability a query term appears in the doc
    # "avg_tf",                           # avg of probability a query term appears in the doc
    "sum_idf",                          # sum of log of inverse document frequency of a query term if it appears in the doc
    # "avg_idf",                          # avg of log of inverse document frequency of a query term if it appears in the doc
    "covered_idf_ratio",                # sum of idf of terms in doc and query divided by sum of idf of terms in query
    "sum_tfidf",                        # sum of product of tf and idf of a query term
    "avg_tfidf",                        # avg of product of tf and idf of a query term
    "bm25",                             # bm25 score
    # "log_bm25",                         # log of bm25 score
    # "sqrt_bm25",                        # sqrt of bm25 score
    # "direchlet_ql",                     # query likelihood score with direchlet smoothing
    # "jelinek_mercer_ql",                # query likelihood score with jelinek-mercer smoothing
    # "kl_wrt_query",                     # kl divergence using query for true distribution and smoothed doc for approx
    # "kl_wrt_doc",                       # kl divergence using doc for true distribution and smoothed query for approx
    "random_noise_10",                  # a random sample on -10 to 10 to detect overfitting
    "random_noise_50",                  # a random sample on -50 to 50 to detect overfitting
]

# return a random permutation such that the first "training_data_split" num entries are stratified in isolation and so are the rest
# the use case is so that when we shuffle using the permutation and slice into two splits, they are independently stratified
def stratified_permutation(feature_data: list[dict], training_data_split: int):
    rel_indices = np.array([
        i for i, features in enumerate(feature_data)
        if features["relevant"]
    ])
    nrel_indices = np.array([
        i for i, features in enumerate(feature_data)
        if not features["relevant"]
    ])

    np.random.shuffle(rel_indices)
    np.random.shuffle(nrel_indices)

    rel_ratio = len(rel_indices) / len(feature_data)

    train_rel_count = round(training_data_split * rel_ratio)
    train_nrel_count = training_data_split - train_rel_count

    train_indices = np.concatenate([
        rel_indices[:train_rel_count],
        nrel_indices[:train_nrel_count],
    ])

    test_indices = np.concatenate([
        rel_indices[train_rel_count:],
        nrel_indices[train_nrel_count:],
    ])

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return np.concatenate([ train_indices, test_indices ])

# for a dataset of n items, choose n items with replacement such that the resulting items have the same label distribution as the input
def stratified_bootstrap(feature_data: list[dict]):
    rel_indices = np.array([
        i for i, features in enumerate(feature_data)
        if features["relevant"]
    ])
    nrel_indices = np.array([
        i for i, features in enumerate(feature_data)
        if not features["relevant"]
    ])

    bootstrap_indices = np.concatenate([
        np.random.choice(rel_indices, size=len(rel_indices), replace=True),
        np.random.choice(nrel_indices, size=len(nrel_indices), replace=True),
    ])

    np.random.shuffle(bootstrap_indices)

    return bootstrap_indices

ridge_alpha = 10000.0
logistic_c = 0.01

# compute, per feature and per layer depth, how well a linear probe recovers the feature
# ridge regression for continuous features, logistic regression for binary features
# recovery is measured in r^2 for continuous, auc for binary
def calculate_r2_progressions(num_layers: int, feature_is_binary: dict[str, bool], training_xs, validation_xs, training_y, validation_y, feature_names = feature_names):
    # store, per layer, a map of feature -> r2/auc value
    training_r2_maps = []
    validation_r2_maps = []

    scaler = StandardScaler()

    for layer_index in range(num_layers):
        training_x = scaler.fit_transform(training_xs[:, layer_index, :])
        validation_x = scaler.transform(validation_xs[:, layer_index, :])

        training_feature_metrics = {}
        validation_feature_metrics = {}

        for feature_index, feature_name in enumerate(feature_names):
            training_y_i = training_y[:, feature_index]
            validation_y_i = validation_y[:, feature_index]

            if feature_is_binary[feature_name]:
                try:
                    # binary target: logistic regression
                    model = LogisticRegression(
                        C=logistic_c,
                        max_iter=200
                    )

                    model.fit(training_x, training_y_i.astype(int))

                    training_predictions = model.predict_proba(training_x)[:, 1]
                    validation_predictions = model.predict_proba(validation_x)[:, 1]

                    training_feature_metrics[feature_name] = roc_auc_score(training_y_i, training_predictions)
                    validation_feature_metrics[feature_name] = roc_auc_score(validation_y_i, validation_predictions)
                except:
                    # some splits have no interlabel variance, such as cross class

                    training_feature_metrics[feature_name] = 0
                    validation_feature_metrics[feature_name] = 0

            else:
                # continuous target: ridge regression
                model = Ridge(alpha=ridge_alpha, solver="auto")
                model.fit(training_x, training_y_i)

                training_predictions = model.predict(training_x)
                validation_predictions = model.predict(validation_x)

                # r2
                training_feature_metrics[feature_name] = r2_score(training_y_i, training_predictions)
                validation_feature_metrics[feature_name] = r2_score(validation_y_i, validation_predictions)

                # pearson
                # training_feature_metrics[feature_name] = pearsonr(training_y_i, training_predictions)[0]
                # validation_feature_metrics[feature_name] = pearsonr(validation_y_i, validation_predictions)[0]

                # spearman
                # training_feature_metrics[feature_name] = spearmanr(training_y_i, training_predictions)[0]
                # validation_feature_metrics[feature_name] = spearmanr(validation_y_i, validation_predictions)[0]

        training_r2_maps.append(training_feature_metrics)
        validation_r2_maps.append(validation_feature_metrics)

    training_r2_progressions = { feature_name: [] for feature_name in feature_names }
    validation_r2_progressions = { feature_name: [] for feature_name in feature_names }

    for feature_r2_map in training_r2_maps:
        for feature_name, r2 in feature_r2_map.items(): training_r2_progressions[feature_name].append(r2)

    for feature_r2_map in validation_r2_maps:
        for feature_name, r2 in feature_r2_map.items(): validation_r2_progressions[feature_name].append(r2)

    return training_r2_progressions, validation_r2_progressions

# single and combined plots of how r2/auc evolves for each feature across layer depth
def plot_r2_progressions(output_path: str, split_name: str, feature_r2_progressions: dict[str, list[float]]):
    os.makedirs(f"{output_path}/{split_name}", exist_ok=True)

    for feature_name, r2_values in feature_r2_progressions.items():
        plt.plot(range(1, len(r2_values) + 1, 1), r2_values)
        plt.xlabel("Layer Index")
        plt.xlim(1, len(r2_values))
        plt.ylabel("R Squared or AUC")
        plt.ylim(0.0, 1.0)
        plt.title(feature_name)
        plt.grid(True)
        plt.savefig(f"{output_path}/{split_name}/{feature_name}.png")
        plt.clf()

    plt.figure(figsize=(16, 10))

    sorted_features = sorted(
        feature_r2_progressions.items(),
        key=lambda item: item[1][-1],
        reverse=True
    )

    for feature_index, (feature_name, r2_values) in enumerate(sorted_features, start=1):
        plt.plot(
            range(1, len(r2_values) + 1, 1),
            r2_values,
            marker="o",
            linewidth=0.5,
            label=f"{feature_index}: {feature_name}"
        )

        plt.text(
            len(r2_values) + 0.5,
            max(r2_values[-1], 0), # ensures rendering fits in tight layout
            str(feature_index),
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center"
        )

    plt.xlabel("Layer Index")
    plt.xlim(1, len(r2_values) + 1)
    plt.ylabel("R Squared or AUC")
    plt.ylim(0.0, 1.0)
    plt.title("Several Classical IR Features")
    plt.grid(True)

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False
    )
    plt.tight_layout()

    plt.savefig(f"{output_path}/{split_name}/all.png")
    plt.close()

# plots with 95% CI of how r2 evolves for each feature across layer depth
def plot_r2_progressions_with_ci(output_path: str, split_name: str, feature_r2_progressions_samples: list[dict[str, list[float]]]):
    os.makedirs(f"{output_path}/{split_name}", exist_ok=True)

    for feature_name in feature_r2_progressions_samples[0]:
        samples = np.array([
            sample[feature_name]
            for sample in feature_r2_progressions_samples
        ])

        means = np.mean(samples, axis=0)
        lower = np.percentile(samples, 2.5, axis=0)
        upper = np.percentile(samples, 97.5, axis=0)

        layers = np.arange(1, len(means) + 1)

        plt.figure(figsize=(10, 6))

        plt.errorbar(
            layers,
            means,
            yerr=[means - lower, upper - means],
            fmt="-o",
            markersize=3,
            capsize=3,
            linewidth=1.5,
        )

        plt.xlabel("Layer Index")
        plt.xlim(1, len(means))
        plt.ylabel("R Squared or AUC")
        plt.ylim(0.0, 1.0)
        plt.title(feature_name)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(f"{output_path}/{split_name}/{feature_name}.png")
        plt.close()

# histograms for distribution of a given feature across relevant, non-relevant, and all pairs
def plot_distributions(
    feature_name: str,
    output_path: str,
    rel_data: list[float],
    nrel_data: list[float],
    all_data: list[float],
    num_bins: int = 30
):
    rel_data = np.asarray(rel_data)
    nrel_data = np.asarray(nrel_data)
    all_data = np.asarray(all_data)

    bins = np.linspace(
        min(all_data.min(), rel_data.min(), nrel_data.min()),
        max(all_data.max(), rel_data.max(), nrel_data.max()),
        num_bins + 1,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    datasets = [
        (rel_data, "Relevant"),
        (nrel_data, "Non-relevant"),
        (all_data, "All"),
    ]

    for ax, (data, title) in zip(axes, datasets):
        ax.hist(data, bins=bins, density=True, alpha=0.7, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel("Density")

    fig.suptitle(f"{feature_name} Distributions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()