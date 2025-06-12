# cage_fusion/benchmarks/moleculenet/run_benchmark.py

import os, sys, torch, numpy as np
import pandas as pd
import shutil, traceback
import deepchem as dc
from transformers import AutoTokenizer, AutoModel
from deepchem.data import NumpyDataset
from deepchem.splits import ScaffoldSplitter
from cage_fusion.configs import get_default_config
from cage_fusion.featurizers import featurize_and_save_streaming
from cage_fusion.models import CAGEFusionModel
from cage_fusion.engine.training import train_model
from cage_fusion.engine.dataset import CageFusionStreamingDataset, MiniBatchCacheDataset
from cage_fusion.engine.data_utils import collate_fn_for_cage_fusion
from cage_fusion.utils.logging_utils import logger
from rich.console import Console
from rich.traceback import install

install()
console = Console()


def compute_token_prior(tokenizer, texts, min_freq=10):
    from collections import Counter

    token_freq = Counter()
    for txt in texts:
        token_freq.update(tokenizer.tokenize(txt))
    token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in token_freq]
    counts = np.array(list(token_freq.values()))
    adj = np.clip(counts, min_freq, None)
    imp = 1.0 / np.sqrt(adj)
    imp /= imp.max()
    prior = np.zeros(tokenizer.vocab_size, dtype=np.float32)
    for tok, i in zip(token_freq.keys(), imp):
        tid = tokenizer.convert_tokens_to_ids(tok)
        prior[tid] = i
    return torch.tensor(prior)


def load_and_split(dataset_name="bace", data_dir="data/molnet"):
    

    tasks, datasets, transformers = getattr(dc.molnet, f"load_{dataset_name}")(
        featurizer=None, split="scaffold", reload=True, data_dir=data_dir
    )
    train_ds, val_ds, test_ds = datasets
    df_train = pd.DataFrame(
        {
            "SMILES_Canonical": train_ds.X,
            **{t: train_ds.y[:, i] for i, t in enumerate(tasks)},
        }
    )
    df_val = pd.DataFrame(
        {
            "SMILES_Canonical": val_ds.X,
            **{t: val_ds.y[:, i] for i, t in enumerate(tasks)},
        }
    )
    df_test = pd.DataFrame(
        {
            "SMILES_Canonical": test_ds.X,
            **{t: test_ds.y[:, i] for i, t in enumerate(tasks)},
        }
    )
    return df_train, df_val, df_test, tasks


def run():
    console.rule("[bold cyan]MoleculeNet Benchmark Pipeline")
    df_train, df_val, df_test, tasks = load_and_split()
    console.log(
        f"Loaded datasets: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}"
    )

    config = get_default_config()
    config["num_tasks"] = len(tasks)
    config["batch_size"] = 16

    tokenizer = AutoTokenizer.from_pretrained(config["model_checkpoint"])
    embedding_model = AutoModel.from_pretrained(config["model_checkpoint"]).eval()
    console.log("Loaded transformer tokenizer & model")

    # Compute token importance prior using training SMILES
    prior = compute_token_prior(tokenizer, df_train.SMILES_Canonical.tolist())
    prior_path = "bench_cache/token_prior.pt"
    os.makedirs("bench_cache", exist_ok=True)
    torch.save(prior, prior_path)
    console.log("Saved token importance prior")

    # Featurize train, val, test
    all_dfs = {"train": df_train, "val": df_val, "test": df_test}
    cache = "bench_cache"
    h5_paths = {}
    graph_paths = {}
    for split, df in all_dfs.items():
        h5, graph_glob, _ = featurize_and_save_streaming(
            df=df,
            name=split,
            label_cols=tasks,
            cache_dir=cache,
            tokenizer=tokenizer,
            model=embedding_model,
            fit_scaler=(split == "train"),
        )
        h5_paths[split], graph_paths[split] = h5, graph_glob

    # Build DataLoaders
    def build_loader(split):
        ds = MiniBatchCacheDataset(
            CageFusionStreamingDataset(
                h5_paths[split], graph_paths[split].replace("*", "0")
            ),
            cache_size=512,
        )
        return torch.utils.data.DataLoader(
            ds, batch_size=config["batch_size"], collate_fn=collate_fn_for_cage_fusion
        )

    train_loader = build_loader("train")
    val_loader = build_loader("val")

    model = CAGEFusionModel(config, token_importance_prior_path=prior_path).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=len(train_loader) * config.get("num_epochs", 5), gamma=1.0
    )

    console.rule("[bold yellow]Training Benchmark")
    try:
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=model.device,
            num_epochs=config.get("num_epochs", 5),
            num_tasks=config["num_tasks"],
            base_cache_dir=cache,
            label_names=tasks,
            tokenizer_obj=tokenizer,
        )
    except Exception:
        logger.error("Benchmark training failed")
        traceback.print_exc()
        sys.exit(1)

    console.log("[bold green]✨ Benchmark complete!")
    console.print(pd.DataFrame(history))


if __name__ == "__main__":
    run()
