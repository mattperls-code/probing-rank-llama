import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

tokenizer_name = "meta-llama/Llama-2-7b"
base_model_name = "meta-llama/Llama-2-7b-hf"
model_name = "castorini/rankllama-v1-7b-lora-passage"

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
            activations[layer_index] = output[0].mean(dim=0).detach().cpu().to(torch.float16)

        return hook

    hooks = [layer.mlp.register_forward_hook(make_hook(layer_index)) for layer_index, layer in enumerate(model.base_model.layers)]

    # TODO: store activations and computed features for all query-doc pairs

    demo_query = "how electric vehicles store and use energy"
    demo_doc = "Electric vehicles (EVs) rely on batteries to store energy, typically lithium-ion cells that provide high energy density and relatively long lifespan. These batteries convert electrical energy into chemical energy during charging and reverse the process during discharging to power the electric motor. Energy management systems regulate charging and discharging rates, ensuring safety and efficiency. Some EVs also use regenerative braking, where kinetic energy is captured during deceleration and converted back into electrical energy to recharge the battery. Understanding battery chemistry, thermal management, and energy flow is crucial for optimizing performance and extending the vehicle's range."

    inputs = tokenizer(
        f"query: {demo_query}, document: {demo_doc}",
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(device)

    with torch.no_grad():
        output = model(**inputs)
        score = output.logits

        print(f"Score: {score}")

        print(f"Activations: {activations[0]}")
        print(f"Shape: {activations[0].shape}")

    for hook in hooks: hook.remove()

if __name__ == "__main__": main()