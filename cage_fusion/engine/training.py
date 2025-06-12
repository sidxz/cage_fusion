import os
import torch
import numpy as np

# Import from other engine modules
from .evaluation import evaluate_model
from .train_epoch import train_one_epoch
from .logging import log_epoch_results, plot_training_history

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    num_epochs=50,
    num_tasks=6,
    base_cache_dir="metric_cache",
    label_names=None,
    tokenizer_obj=None,
    lambda_entropy=0.0,
    lambda_prior=0.01
):
    """
    Full training loop with clean logging, per-epoch checkpointing, 
    and final history visualization.
    """
    checkpoint_path = os.path.join(base_cache_dir, "latest_checkpoint.pt")
    best_model_path = os.path.join(base_cache_dir, "best_model.pt")
    start_epoch = 1
    best_val_auc = -1.0
    
    if os.path.exists(checkpoint_path):
        print(f"✅ Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        best_val_auc = checkpoint.get('best_val_auc', -1.0)
        
        for key in ["scale_graph", "scale_attn", "scale_aux"]:
            if key not in history: history[key] = []
        
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor): state[k] = v.to(device)
    else:
        print("INFO: No checkpoint found. Starting from scratch.")
        history = {
            "train_loss": [], "val_loss": [], "train_mcc": [], "val_mcc": [],
            "train_auc": [], "val_auc": [], "train_pr": [], "val_pr": [],
            "per_task": [], "scale_graph": [], "scale_attn": [], "scale_aux": []
        }

    print(f"🚦 Starting training from epoch {start_epoch} to {num_epochs}")
    print(f"📊 Batches - Train: {len(train_loader)}, Validation: {len(val_loader)}")
    print(f"📈 Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    for epoch in range(start_epoch, num_epochs + 1):
        print(f"\n{'='*25} Epoch {epoch}/{num_epochs} {'='*25}")

        train_cache = os.path.join(base_cache_dir, f"epoch_{epoch}_train")
        train_loss, train_mcc, train_auc, train_pr = train_one_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, num_tasks,
            train_cache, tokenizer_obj, True, lambda_entropy, lambda_prior, label_names
        )
        
        val_cache = os.path.join(base_cache_dir, f"epoch_{epoch}_val")
        val_loss, val_mcc, val_auc, val_pr, _, per_task_metrics = evaluate_model(
            model, val_loader, criterion, device, num_tasks, label_names,
            plot_attn=(epoch % 5 == 0), cache_dir=val_cache, tokenizer_obj=tokenizer_obj
        )
        
        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_mcc"].append(train_mcc); history["val_mcc"].append(val_mcc)
        history["train_auc"].append(train_auc); history["val_auc"].append(val_auc)
        history["train_pr"].append(train_pr); history["val_pr"].append(val_pr)
        history["per_task"].append(per_task_metrics)
        history["scale_graph"].append(model.scale_graph.item())
        history["scale_attn"].append(model.scale_attn.item())
        history["scale_aux"].append(model.scale_aux.item())

        log_epoch_results(epoch, num_epochs, history, label_names, per_task_metrics)
        
        checkpoint_data = {
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'history': history,
            'best_val_auc': best_val_auc
        }
        torch.save(checkpoint_data, checkpoint_path)
        print(f"💾 Checkpoint saved for epoch {epoch} at {checkpoint_path}")

        # Save the best model based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            checkpoint_data['best_val_auc'] = best_val_auc
            torch.save(checkpoint_data, best_model_path)
            print(f"🎉 New best model saved with Val AUC: {best_val_auc:.4f} at {best_model_path}")


    print(f"\n\n{'='*20} Training Complete {'='*20}")
    plot_training_history(history)
    return history
