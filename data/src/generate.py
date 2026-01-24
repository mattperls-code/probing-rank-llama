import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import json
import os
import zarr
from compute_features import compute_features

tokenizer_name = "meta-llama/Llama-2-7b"
base_model_name = "meta-llama/Llama-2-7b-hf"
model_name = "castorini/rankllama-v1-7b-lora-passage"
layer_count = 32
layer_size = 4096

# this dir must contain queries.json, docs.json, and qrels.json
###
# queries.json: Array { query_id: string, content: string }
# docs.json: Array { doc_id: string, content: string }
# qrels.json: Array { query_id: string, rel_doc_ids: string[] }
###
input_path = "../input1"

output_path = "../output1"

def main():
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=1)

    model = PeftModel.from_pretrained(base_model, model_name, torch_dtype=torch.float16)
    model = model.merge_and_unload()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    model.to(device)
    model.eval()

    activations = {}

    def make_hook(layer_index: int):
        def hook(model, input, output):
            activations[layer_index] = output[0].mean(dim=0).detach().cpu().to(torch.float16).numpy()

        return hook

    hooks = [layer.mlp.register_forward_hook(make_hook(layer_index)) for layer_index, layer in enumerate(model.base_model.layers)]

    with open(f"{input_path}/queries.json") as queries_file: queries_json = json.load(queries_file)
    with open(f"{input_path}/docs.json") as docs_file: docs_json = json.load(docs_file)

    os.makedirs(output_path, exist_ok=True)
    
    activation_store = zarr.storage.ZipStore(f"{output_path}/activations.zarr", mode="w")
    activation_db = zarr.create_array(
        store=activation_store,
        shape=(len(queries_json), len(docs_json), layer_count, layer_size),
        chunks=(1, 1, 1, layer_size),
        dtype="f2",
        compressors=zarr.codecs.BloscCodec(
            cname="zstd",
            clevel=3,
            shuffle=zarr.codecs.BloscShuffle.shuffle,
        ),
        overwrite=True
    )

    activation_db.attrs.update({ "model": model_name })

    feature_store = []

    for query_index, query in enumerate(queries_json):
        feature_store.append([])

        for doc_index, doc in enumerate(docs_json):
            query_content = query["content"]
            doc_content = doc["content"]

            inputs = tokenizer( f"query: {query_content}, document: {doc_content}", return_tensors="pt").to(device)

            with torch.no_grad(): model(**inputs)

            for layer_index in range(layer_count): activation_db[query_index, doc_index, layer_index, :] = activations[layer_index]

            features = compute_features(doc_content, query_content)

            feature_store[-1].append(features)

            print(f"Processed (q{query_index}, d{doc_index})")

    activation_store.close()

    with open(f"{output_path}/features.json", "w") as features_file: json.dump(feature_store, features_file, indent=2)

    for hook in hooks: hook.remove()

if __name__ == "__main__": main()