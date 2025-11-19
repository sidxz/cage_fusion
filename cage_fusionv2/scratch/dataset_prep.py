import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from config import MODEL_CONFIG


def load_and_tokenize_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    smiles_col: str = "SMILES",
):
    """
    1. Convert pandas DataFrames to Hugging Face Datasets
    2. Infer label columns from the CSV
    3. Tokenize SMILES strings
    4. Attach multi-label targets under the 'labels' key
    """

    # --- 1. Identify input and label columns ---
    if smiles_col not in train_df.columns:
        raise ValueError(
            f"SMILES column '{smiles_col}' not found in training DataFrame."
        )

    # All non smiles cols are treated as label columns
    label_names = [col for col in train_df.columns if col != smiles_col]
    print("Identified Label names from dataset :", label_names)

    # Create label2id and id2label mappings
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}
    print("Label to ID mapping:", label2id)
    # Ensure SMILES are strings
    train_df[smiles_col] = train_df[smiles_col].astype(str)
    val_df[smiles_col] = val_df[smiles_col].astype(str)
    test_df[smiles_col] = test_df[smiles_col].astype(str)

    # --- 2. Convert to Hugging Face datasets ---
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    raw_datasets = DatasetDict(
        {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        }
    )

    # print(raw_datasets)
    # print("Example row from training set:")
    # print(raw_datasets["train"][0])

    # --- 3. Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["model_ckpt"])

    print(f"Loaded tokenizer from '{MODEL_CONFIG['model_ckpt']}'")
    print("Tokenizer vocab size:", tokenizer.vocab_size)
    print("Model max length:", tokenizer.model_max_length)
    print("Tokenizer special tokens:", tokenizer.special_tokens_map)

    # --- 4. Define tokenize function (batched) ---
    def tokenize(batch):
        # Build [batch_size, num_labels] list of lists for multi-label targets we want for each sample [0,1,...]
        labels = []
        for i in range(len(batch[smiles_col])):
            labels.append([float(batch[name][i]) for name in label_names])

        tokenized = tokenizer(
            batch[smiles_col],
            truncation=True,
        )
        tokenized["labels"] = labels
        return tokenized

    # --- 5. Apply tokenization to all splits ---
    cols_to_remove = raw_datasets["train"].column_names
    tokenized_datasets = raw_datasets.map(
        tokenize,
        batched=True,
        remove_columns=cols_to_remove,
    )

    return tokenized_datasets, label_names, label2id, id2label
