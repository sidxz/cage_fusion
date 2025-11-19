from dataset_prep import load_and_tokenize_datasets
from utils import plot_class_distribution
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

##### PREPARE DATASET AND TOKENIZE #####
TRAIN_CSV_PATH = (
    "/home/sidx/workspace/lab-ai/nuisance_detector/data/balanced-sets-aug-1/train.csv"
)
VAL_CSV_PATH = (
    "/home/sidx/workspace/lab-ai/nuisance_detector/data/balanced-sets-aug-1/val.csv"
)
TEST_CSV_PATH = (
    "/home/sidx/workspace/lab-ai/nuisance_detector/data/balanced-sets-aug-1/test.csv"
)

# train_df = pd.read_csv(TRAIN_CSV_PATH).head(5000)   # for quick testing
# val_df = pd.read_csv(VAL_CSV_PATH).head(1000)
# test_df = pd.read_csv(TEST_CSV_PATH).head(1000)

train_df = pd.read_csv(TRAIN_CSV_PATH)
val_df = pd.read_csv(VAL_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)



tokenized_datasets, label_names, label2id, id2label = load_and_tokenize_datasets(
    train_df, val_df, test_df
)
print("Tokenized datasets:")
print(tokenized_datasets)
# print("Example tokenized input:")
# print(tokenized_datasets["train"][0])
plot_class_distribution(train_df, label_names)

#### LOAD CLASSIFICATION MODEL ####

from transformers import AutoModelForSequenceClassification
from config import MODEL_CONFIG
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_CONFIG["model_ckpt"],
    num_labels=len(label_names),
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id,
).to(device)
print(f"Loaded model from {MODEL_CONFIG['model_ckpt']}")


# Add Data collator
from transformers import DataCollatorWithPadding, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_ckpt"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
from compute_metrics import make_compute_metrics, PrettyPrintCallback

compute_metrics = make_compute_metrics(label_names)


# Trainer args
from transformers import TrainingArguments, Trainer


training_args = TrainingArguments(
    output_dir="./smiles-multilabel-checkpoints",

    # Mixed precision (Ada Lovelace GPUs excel at BF16)
    bf16=True,                   # best choice for RTX 6000 Ada
    fp16=False,                  # don't mix fp16 + bf16
    tf32=True,                   # MUCH faster matmuls on Ada GPU


    learning_rate=5e-5,
    num_train_epochs=25,

    per_device_train_batch_size=128*1,     # try 32 first (Ada 48GB VRAM)
    per_device_eval_batch_size=128*1,

    gradient_accumulation_steps=1,      # increase later if you run out of VRAM

    warmup_ratio=0.1,
    weight_decay=0.01,

    load_best_model_at_end=True,
    metric_for_best_model="roc_auc_macro",
    greater_is_better=True,

    # This chooses the fastest AdamW implementation for your GPU
    optim="adamw_torch_fused",

    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=4,

    # save checkpoints efficiently
    save_total_limit=2,
    
    eval_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[PrettyPrintCallback()],
)

print("Starting training...")
trainer.train()

# ----- Save best model -----
best_model_dir = "./smiles-multilabel-checkpoints/best"
trainer.save_model(best_model_dir)           # saves model + config
tokenizer.save_pretrained(best_model_dir)    # saves tokenizer
print(f"Saved best model to {best_model_dir}")



# ============================
#  VALIDATION THRESHOLD TUNING
# ============================
from utils import find_optimal_thresholds
import numpy as np
print("\n***** Finding optimal thresholds on VALIDATION set *****")
val_output = trainer.predict(tokenized_datasets["validation"])
val_logits = val_output.predictions
val_labels = val_output.label_ids

# Sigmoid to get probabilities
val_probs = 1 / (1 + np.exp(-val_logits))
y_val_true = val_labels.astype(int)

best_thresholds = find_optimal_thresholds(val_probs, y_val_true, label_names)

print("\nBest thresholds per label (validation-tuned):")
for name in label_names:
    print(f"  {name:25s}: {best_thresholds[name]:.3f}")
    
from compute_metrics import compute_all_metrics


# =======================================
# 10. Test evaluation with tuned thresholds
# =======================================

print("\n***** Evaluating on TEST set with tuned thresholds *****")
test_output = trainer.predict(tokenized_datasets["test"])
test_logits = test_output.predictions
test_labels = test_output.label_ids

test_metrics = compute_all_metrics(
    logits=test_logits,
    labels=test_labels,
    label_names=label_names,
    thresholds=best_thresholds,  # <- use tuned thresholds
)

print("\nTest metrics (tuned thresholds):")
for k, v in sorted(test_metrics.items()):
    if isinstance(v, float):
        print(f"{k:30s}: {v:.6f}")
    else:
        print(f"{k:30s}: {v}")

# -------------------------
# 11. Confusion matrices on test set
# -------------------------
from utils import plot_multilabel_confusion
print("\nPlotting confusion matrices for TEST set...")
test_probs = 1 / (1 + np.exp(-test_logits))
thr_vec = np.array([best_thresholds[name] for name in label_names])[None, :]
y_test_pred = (test_probs >= thr_vec).astype(int)
y_test_true = test_labels.astype(int)

plot_multilabel_confusion(
    y_true=y_test_true,
    y_pred=y_test_pred,
    label_names=label_names,
    save_path="confusion_matrices_test.png",
)