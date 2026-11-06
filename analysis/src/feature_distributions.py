import os
import json

from util import plot_distributions

data_path = "../../data/output-3k"
output_path = "../distribution-results"

if __name__ == "__main__":
    os.makedirs(output_path, exist_ok=True)

    with open(f"{data_path}/features.json", "r") as features_file: feature_data = json.load(features_file)

    rel_feature_data = feature_data[:500]
    nrel_feature_data = feature_data[500:]

    for feature_name, example_feature_value in feature_data[0].items():
        if isinstance(example_feature_value, bool): continue

        all_data = [ features[feature_name] for features in feature_data ]
        rel_data = [ features[feature_name] for features in rel_feature_data ]
        nrel_data = [ features[feature_name] for features in nrel_feature_data ]

        plot_distributions(
            feature_name,
            f"{output_path}/{feature_name}.png",
            rel_data,
            nrel_data,
            all_data,
        )