"""cage_fusion.inference — pipeline and gradient-saliency explainer."""

from .pipeline import CageFusionPipeline, predict_smiles, predict_and_explain
from .explainer import GradientExplainer, explain_smiles

__all__ = [
    "CageFusionPipeline",
    "predict_smiles",
    "predict_and_explain",
    "GradientExplainer",
    "explain_smiles",
]
