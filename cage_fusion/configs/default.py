import torch


def get_default_config():
    """
    Returns the default hyperparameter configuration for the CAGEFusionModel.

    This configuration includes:
    - Model architecture hyperparameters
    - Training setup details
    - Regularization parameters
    - Tokenizer and data processing settings
    - Device allocation logic based on CUDA availability
    """
    try:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as e:
        # Fallback to CPU in case of unexpected error when checking CUDA
        device_type = "cpu"

    config = {
        # === Model Hyperparameters ===
        "graph_dim": 300,  # Dimension of graph features
        "embedding_dim": 768,  # Token embedding size
        "aux_feature_dim": 217,  # Dimension of auxiliary features
        "num_tasks": 6,  # Number of prediction tasks
        "num_heads": 8,  # Number of attention heads
        "cross_attn_dropout": 0.1,  # Dropout rate in cross-attention
        "proj_dropout": 0.1,  # Dropout rate in projection layers
        "graph_only_mode": False,  # Use only graph features without token embeddings
        "use_co_attention": True,  # Use Co-Attention mechanism
        "co_attention_layers": 2,  # Number of Co-Attention layers
        "fusion_dropout_1": 0.3,  # Dropout rate in first fusion layer
        "fusion_dropout_2": 0.2,  # Dropout rate in second
        # === Training Hyperparameters ===
        "learning_rate": 3e-4,  # Learning rate for the optimizer
        "num_epochs": 50,  # Total number of training epochs
        "batch_size": 192,  # Number of samples per batch
        "warmup_fraction": 0.2,  # Warmup steps as a fraction of total steps
        "clip_grad_norm": 1.0,  # Gradient clipping to prevent exploding gradients
        # Scaled Attention Factor
        "scaled_graph_factor": 1.0,  # Scaling factor for attention scores
        "scale_attn_factor": 0.5,  # Scaling factor for attention scores
        "scale_aux_factor": 0.1,  # Scaling factor for auxiliary features
        # === Attention Regularization ===
        "lambda_entropy": 0.001,  # Entropy loss weight for attention sparsity
        "lambda_prior": 0.000,  # Prior loss weight using token importance
        # === Data Processing Settings ===
        "max_seq_len": 512,  # Maximum sequence length for token inputs
        "neg_to_pos_ratio": 3,  # Ratio of negative to positive samples in training
        # === Tokenizer & Model Checkpoint ===
        "model_checkpoint": "unikei/bert-base-smiles",  # Pretrained tokenizer/model checkpoint
        # === Execution Device ===
        "device": device_type,  # Automatically assigned device based on CUDA availability
        # === Optional Settings ===
        "token_importance_prior": None,  # Path to pre-computed token importance (optional)
    }

    return config
