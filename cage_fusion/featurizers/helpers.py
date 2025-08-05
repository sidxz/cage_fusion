import joblib
import h5py
import numpy as np
import torch
from tqdm import tqdm
from cage_fusion.utils.logging_utils import logger


def initialize_hdf5_file(h5_path, num_samples, seq_len, embed_dim, aux_dim, num_labels):
    """
    Initializes an HDF5 file to store embeddings, token IDs, features, and labels.

    Args:
        h5_path (str): Path to the HDF5 file.
        num_samples (int): Total number of samples.
        seq_len (int): Max sequence length per molecule.
        embed_dim (int): Token embedding dimensionality.
        aux_dim (int): Auxiliary features dimension.
        num_labels (int): Number of prediction tasks.
    """
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "embedding", shape=(num_samples, seq_len, embed_dim), dtype=np.float32
        )
        f.create_dataset("input_ids", shape=(num_samples, seq_len), dtype=np.int32)
        f.create_dataset(
            "auxiliary_features", shape=(num_samples, aux_dim), dtype=np.float32
        )
        # f.create_dataset(
        #     "auxiliary_features_normalized", shape=(num_samples, aux_dim), dtype=np.float32
        # )
        f.create_dataset("labels", shape=(num_samples, num_labels), dtype=np.float32)
        f.create_dataset("original_indices", shape=(num_samples,), dtype=np.int64)
        string_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("smiles", shape=(num_samples,), dtype=string_dt, chunks=True)
    logger.info(f"Initialized HDF5 file at {h5_path}")


def featurize_batch(tokenizer, model, smiles_batch, seq_len, device, vocab_size):
    """
    Tokenizes SMILES and extracts embeddings using a pretrained transformer model.

    Args:
        tokenizer: HuggingFace tokenizer.
        model: HuggingFace model.
        smiles_batch (List[str]): List of SMILES strings.
        seq_len (int): Max sequence length.
        device (torch.device): Target device.
        vocab_size (int): Maximum token ID allowed.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Token IDs and embedding tensors (on CPU).
    """
    # logger.info(f"Featurizing {len(smiles_batch)} SMILES strings...")
    inputs = tokenizer(
        smiles_batch,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    input_ids = inputs["input_ids"]

    if (input_ids >= vocab_size).any() or (input_ids < 0).any():
        raise ValueError("Token IDs contain values out of allowed bounds.")

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = model(**inputs)
        embeddings = output.last_hidden_state

    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        raise ValueError("Embeddings contain NaN or Inf.")
    # logger.info("Featurization complete.")
    return input_ids.cpu().numpy(), embeddings.cpu().numpy()


def process_auxiliary_features(
    batch_df,
    i_offset,
    graph_feats,
    graph_featurizer,
    desc_calc,
    label_cols,
    scaler,
    h5_file,
    fit_scaler=True,
    clean_descriptors=lambda x: x,
):
    """
    Featurizes and stores auxiliary features and labels into HDF5.

    Args:
        batch_df (pd.DataFrame): Batch of molecules and labels.
        i_offset (int): Offset index in the dataset.
        graph_feats (List): List to append graph features.
        graph_featurizer (callable): Function to convert molecule to graph.
        desc_calc (DescriptorCalculator): Calculator for auxiliary descriptors.
        label_cols (List[str]): List of label column names.
        scaler (StandardScaler): Scaler for feature normalization.
        h5_file (h5py.File): Open HDF5 file in write mode.
        fit_scaler (bool): If True, update scaler online.
        clean_descriptors (callable): Optional function to clean descriptors.

    Returns:
        List: Updated graph_feats list.
    """
    # logger.info(f"Processing auxiliary features batch of size {len(batch_df)}")
    batch_aux = []
    for j, row in batch_df.iterrows():
        idx = i_offset + (j - batch_df.index[0])
        # h5_file["original_indices"][idx] = row["original_index"]
        mol = row["mol"]
        graph_feats.append(graph_featurizer(mol))

        desc = clean_descriptors(np.array(desc_calc.CalcDescriptors(mol)))
        h5_file["auxiliary_features"][idx] = desc

        # Robust label handling for inference
        labels_arr = np.zeros(len(label_cols), dtype=np.float32)
        missing = []
        for i, col in enumerate(label_cols):
            if col in row:
                labels_arr[i] = row[col]
            else:
                missing.append(col)
        if missing:
            logger.debug(
                f"Missing label columns for prediction: {missing} (filling zeros)"
            )

        h5_file["labels"][idx] = labels_arr
        batch_aux.append(desc)

    if fit_scaler and batch_aux:
        scaler.partial_fit(np.array(batch_aux))
    # logger.info("Auxiliary features processed.")
    return graph_feats


def save_graph_features(graph_feats, base_path, part_idx):
    """
    Saves a batch of graph features to a compressed joblib file.

    Args:
        graph_feats (List): Graph features list.
        base_path (str): Base path for graph files.
        part_idx (int): Part number for naming.
    """
    path = f"{base_path}_{part_idx}.pkl"
    joblib.dump(graph_feats, path, compress=3)
    logger.info(f"Graph features saved to {path}")


def normalize_auxiliary_features(
    h5_path, scaler, aux_dim, batch_size=512, name="features"
):
    """
    Normalizes stored auxiliary features using a fitted scaler.

    Args:
        h5_path (str): Path to the HDF5 file.
        scaler (StandardScaler): Fitted scaler.
        aux_dim (int): Dimensionality of auxiliary features.
        batch_size (int): Batch size for normalization.
        name (str): Name label for logging.
    """
    logger.info(f"Normalizing auxiliary features in {h5_path} ...")
    with h5py.File(h5_path, "a") as f:
        N = f["auxiliary_features"].shape[0]

        if "auxiliary_features_normalized" in f:
            del f["auxiliary_features_normalized"]
        f.create_dataset(
            "auxiliary_features_normalized", shape=(N, aux_dim), dtype=np.float32
        )

        for i in tqdm(range(0, N, batch_size), desc=f"Normalizing {name}"):
            batch = f["auxiliary_features"][i : i + batch_size]
            f["auxiliary_features_normalized"][i : i + batch_size] = scaler.transform(
                batch
            )

        del f["auxiliary_features"]
        logger.info(f"Removed original 'auxiliary_features' dataset to conserve space.")

        logger.info(f"Auxiliary features normalization complete for '{name}'")
