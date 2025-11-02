import torch

def load_partial_weights(model, checkpoint_path):
    """
    Loads selected weights from a checkpoint, skipping fusion/output layers.
    """
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device_type, weights_only=False)
    pretrained_dict = checkpoint["model_state_dict"]
    model_dict = model.state_dict()

    include_keys = [
        "message_passing.",
        "global_aggregation.",
        "encoder.predictor.",
        "graph_proj.",
        "embedding_proj.",
        "cross_attn.",
        "co_attn.",
        "gate_graph.",
        "gate_embedding.",
        "attention_norm_layers.",
        "ffn_norm_layers.",
        "cross_attn_ffn.",
        "fg_prompter.",
    ]

    filtered_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if any(k.startswith(prefix) for prefix in include_keys)
    }

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    print(f"✅ Loaded {len(filtered_dict)} parameters from checkpoint.")
