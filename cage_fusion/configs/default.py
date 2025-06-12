import torch

def get_default_config():
    """Returns the default hyperparameter configuration for the CAGEFusionModel."""
    config = {
        # Model hyperparameters
        "graph_dim": 300,
        "embedding_dim": 768,
        "aux_feature_dim": 217,
        "num_tasks": 6,
        "num_heads": 8,
        "cross_attn_dropout": 0.2,
        "proj_dropout": 0.1,
        "use_atom_level_queries": True,
        "use_advanced_features": True,

        # Training hyperparameters
        "learning_rate": 3e-4,
        "num_epochs": 50,
        "batch_size": 192,
        "warmup_fraction": 0.2,
        "clip_grad_norm": 1.0,

        # Regularization hyperparameters for the attention mechanism
        "lambda_entropy": 0.001,  # Encourages the model to spread its attention
        "lambda_prior": 0.01,     # Guides attention using pre-computed token importance

        # Data processing
        "max_seq_len": 512,
        "neg_to_pos_ratio": 3,
        
        # Tokenizer settings
        "model_checkpoint": "unikei/bert-base-smiles",
        
        # Device
        "device": "cuda" if torch.cuda.is_available() else "cpu",

        # Optional: Path to a pre-computed token importance tensor
        "token_importance_prior": None,
    }
    return config
