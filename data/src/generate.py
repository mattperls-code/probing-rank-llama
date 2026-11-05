from collection_statistics import collection
from itertools import islice
import random
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import json
import os
import zarr
from zarr.storage import LocalStore
from compute_features import compute_features

num_rel_pairs = 500
num_nrel_pairs = 2500

tokenizer_name = "meta-llama/Llama-2-7b"
base_model_name = "meta-llama/Llama-2-7b-hf"
model_name = "castorini/rankllama-v1-7b-lora-passage"
layer_count = 32
layer_size = 4096

output_path = "../output-3k-new"

def main():
    print("Sampling query-doc pairs")

    docs_store = collection.docs_store()

    queries = { query.query_id: query.text for query in collection.queries_iter() }

    # stored as (query_text, doc_id) since docs have efficient retrieval and expensive queries dict gets GC sooner
    query_doc_pairs = []

    for qrel in islice(collection.qrels_iter(), num_rel_pairs): query_doc_pairs.append((queries[qrel.query_id], qrel.doc_id))
    
    random.seed(3.14159)

    # assume negligible collision rate
    for query in islice(collection.queries_iter(), num_nrel_pairs): query_doc_pairs.append((queries[query.query_id], str(random.randint(0, 100000))))
    
    del queries

    print("Finished selecting pairs")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using Device: {device}")

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        model_name,
    )

    model = model.merge_and_unload()

    model.to(device)
    model.eval()

    activations = {}

    def make_hook(layer_index: int):
        def hook(model, input, output):
            activations[layer_index] = output[0].mean(dim=0).detach().cpu().to(torch.float16).numpy()

        return hook

    hooks = [
        layer.mlp.register_forward_hook(make_hook(layer_index))
        for layer_index, layer in enumerate(model.base_model.layers)
    ]

    os.makedirs(output_path, exist_ok=True)
    
    activation_store = LocalStore(f"{output_path}/activations.zarr")
    activation_data = zarr.create_array(
        store=activation_store,
        shape=(len(query_doc_pairs), layer_count, layer_size),
        chunks=(1, 1, layer_size),
        dtype="f2",
        compressors=[],
        overwrite=True
    )

    activation_data.attrs.update({ "model": model_name })

    feature_data = []

    print("Beginning activation extraction")

    for pair_index, (query_text, doc_id) in enumerate(query_doc_pairs):
        doc_text = docs_store.get(doc_id).text

        inputs = tokenizer(f"query: {query_text}, document: {doc_text}", return_tensors="pt").to(device)

        # TODO: this could be done in parallel with better hooks implementation
        with torch.no_grad(): score = model(**inputs).logits.item()

        print(score)

        for layer_index in range(layer_count): activation_data[pair_index, layer_index, :] = activations[layer_index]

        features = compute_features(doc_text, query_text)

        features["score"] = score

        # manually add the relevance label
        features["relevant"] = pair_index < num_rel_pairs

        # random class assignment to check overfitting. balanced and weighted by real split
        features["random_class"] = random.random() < 0.5
        features["weighted_random_class"] = random.random() < num_rel_pairs / (num_rel_pairs + num_nrel_pairs)

        feature_data.append(features)

        print(f"Processed pair {pair_index+1}/{len(query_doc_pairs)}")

    activation_store.close()

    with open(f"{output_path}/features.json", "w") as features_file: json.dump(feature_data, features_file, indent=2)

    for hook in hooks: hook.remove()

if __name__ == "__main__": main()