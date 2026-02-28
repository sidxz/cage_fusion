"""
cage_fusion
===========
Multimodal molecular property prediction — HuggingFace-inspired API.

Package layout
--------------
::

    cage_fusion
    ├── configuration/   CageFusionConfig
    ├── modeling/        CAGEFusionModel, task heads, MolGraphEncoder
    ├── data/            CageFusionStreamingDataset, collate_cage_fusion
    ├── featurization/   featurize_and_save_streaming
    ├── chemistry/       get_functional_groups
    ├── training/        Trainer, TrainingArguments
    ├── evaluation/      evaluate_model
    ├── inference/       CageFusionPipeline, GradientExplainer
    ├── visualization/   attention maps, functional-group plots
    ├── auto.py          AutoCageFusion — automatic task-head dispatch
    └── utils/           logging, device helpers, HF loader

Quick start
-----------
**Predict with a pre-trained model**::

    from cage_fusion import CageFusionPipeline

    pipe = CageFusionPipeline.from_pretrained("checkpoints/my_run")

    # Single SMILES → dict
    result = pipe("CC(=O)Oc1ccccc1C(=O)O")

    # List of SMILES → list of dicts
    results = pipe(["SMILES1", "SMILES2", "SMILES3"])

    # DataFrame → DataFrame
    import pandas as pd
    df_out = pipe(pd.DataFrame({"SMILES": ["CC(=O)Oc1ccccc1C(=O)O"]}))

**Build a classification model**::

    from cage_fusion import CageFusionConfig, CAGEFusionForMultiLabelClassification

    config = CageFusionConfig(
        num_labels=4,
        model_task="classification",
        label_names=["PAINS_A", "PAINS_B", "Aggregator", "Chelator"],
    )
    model = CAGEFusionForMultiLabelClassification(config)

**Automatic task-head dispatch**::

    from cage_fusion import AutoCageFusion, CageFusionConfig

    # From a config object
    config = CageFusionConfig(num_labels=12, model_task="regression")
    model  = AutoCageFusion.from_config(config)

    # From a saved checkpoint
    model  = AutoCageFusion.from_pretrained("checkpoints/my_run")

**Gradient saliency**::

    from cage_fusion import CageFusionPipeline
    from cage_fusion.inference import GradientExplainer

    pipe = CageFusionPipeline.from_pretrained("checkpoints/my_run")
    exp  = GradientExplainer(pipe)
    out  = exp.explain("CC(=O)Oc1ccccc1C(=O)O", target_task="PAINS_A")
    # out["tokens"], out["token_saliency"], out["aux_saliency"]
"""

__version__ = "0.2.0"

# ── Configuration ──────────────────────────────────────────────────────────
from .configuration import CageFusionConfig

# ── Models ─────────────────────────────────────────────────────────────────
from .modeling import (
    CAGEFusionModel,
    CAGEFusionForMultiLabelClassification,
    CAGEFusionForRegression,
    CAGEFusionPreTrainedModel,
)

# ── Automatic task-head dispatch ───────────────────────────────────────────
from .auto import AutoCageFusion

# ── Inference ──────────────────────────────────────────────────────────────
from .inference import (
    CageFusionPipeline,
    GradientExplainer,
    predict_smiles,
    predict_and_explain,
    explain_smiles,
)

# ── Evaluation ─────────────────────────────────────────────────────────────
from .evaluation import evaluate_model

__all__ = [
    # version
    "__version__",
    # config
    "CageFusionConfig",
    # models
    "CAGEFusionModel",
    "CAGEFusionForMultiLabelClassification",
    "CAGEFusionForRegression",
    "CAGEFusionPreTrainedModel",
    # auto dispatch
    "AutoCageFusion",
    # inference
    "CageFusionPipeline",
    "GradientExplainer",
    "predict_smiles",
    "predict_and_explain",
    "explain_smiles",
    # evaluation
    "evaluate_model",
]
