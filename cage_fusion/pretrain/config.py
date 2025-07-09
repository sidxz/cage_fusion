def get_config():
    config = {
        "dataset_path": "data/pretrain/250k_rndm_zinc_drugs_clean_3.csv",
        "batch_size": 256,
        "num_epochs": 100,
        "lr": 1e-4,
        "temperature": 0.1,
        "output_dir": "pretrained_weights",
        "graph_dim": 300,  # or  hidden_dim used in MPNN layers
        "num_tasks": 1,
    }
    return config
