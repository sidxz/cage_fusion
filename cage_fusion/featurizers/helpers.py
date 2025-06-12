import joblib
import h5py
import numpy as np
import torch
from tqdm import tqdm
from cage_fusion.utils.logging_utils import logger


def initialize_hdf5_file(h5_path, N, D_seq_len, D_embedding, D_aux_feats, L):
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            "embedding", shape=(N, D_seq_len, D_embedding), dtype=np.float32
        )
        f.create_dataset("input_ids", shape=(N, D_seq_len), dtype=np.int32)
        f.create_dataset("auxiliary_features", shape=(N, D_aux_feats), dtype=np.float32)
        f.create_dataset("labels", shape=(N, L), dtype=np.float32)


def featurize_batch(tokenizer, model, smiles_batch, D_seq_len, device, vocab_size):
    inputs = tokenizer(
        smiles_batch,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=D_seq_len,
    )
    input_ids = inputs["input_ids"]

    if (input_ids >= vocab_size).any() or (input_ids < 0).any():
        raise ValueError("Token IDs out of bounds")

    with torch.no_grad():
        output = model(**{k: v.to(device) for k, v in inputs.items()})
        embeddings = output.last_hidden_state

    if torch.isnan(embeddings).any() or torch.isinf(embeddings).any():
        raise ValueError("NaN or Inf in model output")

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
    fit_scaler,
    clean_descriptors,
):
    batch_aux = []
    for j, row in batch_df.iterrows():
        idx = i_offset + (j - batch_df.index[0])
        graph_feats.append(graph_featurizer(row["mol"]))

        desc = clean_descriptors(np.array(desc_calc.CalcDescriptors(row["mol"])))
        h5_file["auxiliary_features"][idx] = desc
        h5_file["labels"][idx] = row[label_cols].astype(np.float32)
        batch_aux.append(desc)

    if fit_scaler and batch_aux:
        scaler.partial_fit(np.array(batch_aux))

    return graph_feats


def save_graph_features(graph_feats, graph_path_base, part_idx):
    path = f"{graph_path_base}_{part_idx}.pkl"
    joblib.dump(graph_feats, path, compress=3)
    logger.info(f"Saved graph features to {path}")


def normalize_auxiliary_features(h5_path, scaler, D_aux_feats, batch_size, name):
    with h5py.File(h5_path, "a") as f:
        N = f["auxiliary_features"].shape[0]
        if "auxiliary_features_normalized" in f:
            del f["auxiliary_features_normalized"]
        f.create_dataset(
            "auxiliary_features_normalized", shape=(N, D_aux_feats), dtype=np.float32
        )

        for i in tqdm(range(0, N, batch_size), desc=f"Normalizing {name}"):
            batch = f["auxiliary_features"][i : i + batch_size]
            f["auxiliary_features_normalized"][i : i + batch_size] = scaler.transform(
                batch
            )

        logger.info(f"Normalization complete for '{name}'")
