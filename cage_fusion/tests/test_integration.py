import os
import sys
import torch
import pandas as pd
import shutil
import traceback
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.optim as optim

# Add the project root to the Python path to allow for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import all necessary components directly from our library
from cage_fusion.configs import get_default_config
from cage_fusion.engine.dataset import CageFusionStreamingDataset, MiniBatchCacheDataset
from cage_fusion.engine.training import train_model 
from cage_fusion.engine.utils import collate_fn_for_cage_fusion
from cage_fusion.featurizers import featurize_and_save_streaming
from cage_fusion.models import CAGEFusionModel

def run_library_test():
    """
    A self-contained integration test for the cage_fusion library.
    It uses dummy data to run the entire training pipeline for 5 epochs.
    """
    print("--- 🧪 Starting CAGE Fusion Library Integration Test ---")
    
    # --- 1. Setup Paths and Dummy Data ---
    print("\n--- [Step 1/7] Setting up test environment ---")
    test_dir = os.path.dirname(__file__)
    data_path = os.path.join(test_dir, "dummy_data.csv")
    cache_dir = os.path.join(test_dir, "test_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    labels = ["Task_1", "Task_2"]
    print("✅ Environment setup complete.")
    
    # --- 2. Load Config and Models ---
    print("\n--- [Step 2/7] Loading configuration and pre-trained models ---")
    config = get_default_config()
    # Override config for a quick test run
    config['num_tasks'] = 2
    config['batch_size'] = 2
    config['num_epochs'] = 5
    
    # Log the configuration being used
    print("  [Test Config] Epochs: 5, Batch Size: 2, Tasks: 2")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model_checkpoint'])
    embedding_model = AutoModel.from_pretrained(config['model_checkpoint'])
    print("✅ Config and pre-trained models loaded.")
    
    # --- 3. Featurization Step ---
    print("\n--- [Step 3/7] Testing Featurization Pipeline ---")
    h5_path, _, _ = featurize_and_save_streaming(
        df=df, name="dummy", label_cols=labels, cache_dir=cache_dir,
        tokenizer=tokenizer, model=embedding_model, fit_scaler=True
    )
    print("✅ Featurization complete.")
    
    # --- 4. DataLoader Step ---
    print("\n--- [Step 4/7] Testing Dataset and DataLoader creation ---")
    graph_path = os.path.join(cache_dir, "dummy_graph_feats_part_0.pkl")
    # Use the same featurized data for both train and validation for this test
    # CORRECTED: Wrap the streaming dataset with the LRU cache
    mini_cache_size = 4 # Use a small cache for the test
    print(f"  [Test Config] Using MiniBatchCacheDataset with size {mini_cache_size}")
    train_dataset = MiniBatchCacheDataset(
        CageFusionStreamingDataset(h5_path, graph_path),
        cache_size=mini_cache_size
    )
    val_dataset = MiniBatchCacheDataset(
        CageFusionStreamingDataset(h5_path, graph_path),
        cache_size=mini_cache_size
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], collate_fn=collate_fn_for_cage_fusion)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], collate_fn=collate_fn_for_cage_fusion)
    print("✅ Train and Val DataLoaders created successfully.")
    
    # --- 5. Model Initialization Step ---
    print("\n--- [Step 5/7] Testing Model Initialization ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAGEFusionModel(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=total_steps, gamma=1.0) # Dummy scheduler
    print(f"✅ Model initialized on device: {device}")
    
    # --- 6. Full Training Loop Test ---
    print(f"\n--- [Step 6/7] Testing Full `train_model` Loop ({config['num_epochs']} epochs) ---")
    try:
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            num_epochs=config['num_epochs'],
            num_tasks=config['num_tasks'],
            base_cache_dir=cache_dir,
            label_names=labels,
            tokenizer_obj=tokenizer
        )
        print(f"\n✅ `train_model` completed without errors.")
        assert "train_loss" in history and len(history["train_loss"]) == config['num_epochs']
        assert "val_loss" in history and len(history["val_loss"]) == config['num_epochs']
        print("✅ History object seems valid.")

    except Exception:
        print(f"🚨 FAILED during training loop. Full traceback below:")
        traceback.print_exc()
        shutil.rmtree(cache_dir)
        return

    # --- 7. Cleanup ---
    print("\n--- [Step 7/7] Cleaning up cache directory ---")
    shutil.rmtree(cache_dir)
    print("\n🎉 Library Integration Test Passed Successfully! 🎉")


if __name__ == "__main__":
    run_library_test()
