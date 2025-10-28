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
        "attn_mode": "self_graph",  # Attention mode: 'cross', 'self_tokens', 'self_graph' or 'self_both'
        "graph_dim": 300,  # Dimension of graph features
        # "embedding_dim": 768,  # Token embedding size -> unikei/bert-base-smiles
        "embedding_dim": 384,  # Token embedding size -> DeepChem/ChemBERTa-77M-MTR
        "aux_feature_dim": 217,  # Dimension of auxiliary features
        "num_tasks": 4,  # Number of prediction tasks
        "num_heads": 8,  # Number of attention heads
        "cross_attn_dropout": 0.16,  # Dropout rate in cross-attention
        "proj_dropout": 0.06,  # Dropout rate in projection layers
        # === Model Architecture Settings ===
        "use_co_attention": True,  # Use Co-Attention mechanism
        "use_aux_features": True,  # Use auxiliary features in the model
        "use_fg_prompt": True,  # Use functional group prompts
        "co_attention_layers": 1,  # Number of Co-Attention layers
        # === Scaled Attention Factor ===
        "scaled_graph_factor": 1.0,  # Scaling factor for attention scores
        "scale_attn_factor": 1.0,  # Scaling factor for attention scores
        "scale_aux_factor": 1.0,  # Scaling factor for auxiliary features
        "scaled_fg_factor": 0.5,  # Scaling factor for functional group prompts
        # === Fusion Layer Settings ===
        "fusion_dropout_1": 0.01,  # Dropout rate in first fusion layer
        "fusion_dropout_2": 0.01,  # Dropout rate in second
        # === Training Hyperparameters ===
        "learning_rate": 0.001,  # Learning rate for the optimizer
        "num_epochs": 50,  # Total number of training epochs
        "batch_size": 128,  # Number of samples per batch
        "warmup_fraction": 0.09,  # Warmup steps as a fraction of total steps
        # === Attention Regularization ===
        "lambda_entropy": 0.00,  # Entropy loss weight for attention sparsity
        "lambda_prior": 0.000,  # Prior loss weight using token importance
        # === Data Processing Settings ===
        # === Tokenizer & Model Checkpoint ===
        # "model_checkpoint": "unikei/bert-base-smiles",  # Pretrained tokenizer/model checkpoint
        "model_checkpoint": "DeepChem/ChemBERTa-77M-MTR",  # Pretrained tokenizer/model checkpoint
        # === Execution Device ===
        "device": device_type,  # Automatically assigned device based on CUDA availability
        # === Optional Settings ===
        "token_importance_prior": None,  # Path to pre-computed token importance (optional)
        "resume_with_new_arch": False,
    }

    return config
