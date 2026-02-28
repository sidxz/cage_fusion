"""
cage_fusion/auto.py
====================
Automatic model dispatch — similar to HuggingFace's ``AutoModel`` family.

:class:`AutoCageFusion` inspects ``config.model_task`` and returns the
appropriate task head without requiring the caller to import the concrete
class.

Quick start
-----------
**Load from a saved checkpoint** (most common)::

    from cage_fusion import AutoCageFusion

    model = AutoCageFusion.from_pretrained("checkpoints/my_run")
    # Returns CAGEFusionForMultiLabelClassification or CAGEFusionForRegression
    # depending on the config stored in the checkpoint directory.

**Construct from a config object** (for training)::

    from cage_fusion import AutoCageFusion, CageFusionConfig

    config = CageFusionConfig(
        num_labels=12,
        model_task="regression",
        label_names=["logP", "logD", "solubility"],
    )
    model = AutoCageFusion.from_config(config)
    # -> CAGEFusionForRegression(config)
"""

from __future__ import annotations

from typing import Optional

from cage_fusion.configuration.configuration_cage import CageFusionConfig
from cage_fusion.utils.hf_loader import _resolve_pretrained_path
from cage_fusion.modeling.modeling_cage import (
    CAGEFusionForMultiLabelClassification,
    CAGEFusionForRegression,
    CAGEFusionPreTrainedModel,
)

# Registry: model_task -> model class
_TASK_TO_CLASS = {
    "classification": CAGEFusionForMultiLabelClassification,
    "regression": CAGEFusionForRegression,
}


class AutoCageFusion:
    """
    Factory that instantiates the correct CAGEFusion task head.

    ``AutoCageFusion`` is not meant to be instantiated directly — use the
    class-methods :py:meth:`from_config` and :py:meth:`from_pretrained`.

    Supported ``model_task`` values
    --------------------------------
    - ``"classification"`` → :class:`~cage_fusion.modeling.CAGEFusionForMultiLabelClassification`
    - ``"regression"``     → :class:`~cage_fusion.modeling.CAGEFusionForRegression`
    """

    # Prevent instantiation
    def __init_subclass__(cls, **kwargs):  # noqa: D105
        super().__init_subclass__(**kwargs)

    def __new__(cls, *args, **kwargs):  # noqa: D105
        raise TypeError(
            "AutoCageFusion is a factory class and cannot be instantiated. "
            "Use AutoCageFusion.from_config(config) or "
            "AutoCageFusion.from_pretrained(path)."
        )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: CageFusionConfig) -> CAGEFusionPreTrainedModel:
        """
        Instantiate a fresh model from a config object.

        Args:
            config: A :class:`~cage_fusion.configuration.CageFusionConfig`
                with ``model_task`` set to ``"classification"`` or
                ``"regression"``.

        Returns:
            An untrained :class:`~cage_fusion.modeling.CAGEFusionPreTrainedModel`
            sub-class matching ``config.model_task``.

        Raises:
            ValueError: If ``config.model_task`` is not a recognised task.

        Example::

            config = CageFusionConfig(num_labels=4, model_task="classification")
            model  = AutoCageFusion.from_config(config)
        """
        model_cls = _TASK_TO_CLASS.get(config.model_task)
        if model_cls is None:
            raise ValueError(
                f"Unknown model_task '{config.model_task}'. "
                f"Supported: {list(_TASK_TO_CLASS)}"
            )
        return model_cls(config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Optional[CageFusionConfig] = None,
        strict: bool = True,
        load_backbone_only: bool = False,
        **kwargs,
    ) -> CAGEFusionPreTrainedModel:
        """
        Load a model from a checkpoint directory or HuggingFace Hub repo.

        If *config* is not provided it is read from ``config.json`` in the
        checkpoint directory.

        Args:
            pretrained_model_name_or_path: Local directory **or** a HuggingFace
                Hub repo ID (e.g. ``"sidxz/cage-fusion-nuisance"``).  Hub repos
                are downloaded on first call and cached locally.
            config: Optional :class:`~cage_fusion.configuration.CageFusionConfig`
                that overrides the one stored in the checkpoint.  Required when
                *load_backbone_only=True* and the new task has different
                ``num_labels``.
            strict: Passed to :py:meth:`load_state_dict`; set to ``False``
                when transferring weights across architectures.
            load_backbone_only: When ``True``, forces ``strict=False`` and logs
                a clear message that only the encoder weights will be restored
                while the task head is randomly initialised.  Use this for
                transfer learning to a new task.
            **kwargs: Additional keyword arguments forwarded to the model
                ``from_pretrained`` classmethod.

        Returns:
            Loaded model (weights restored from the checkpoint).

        Raises:
            FileNotFoundError: If ``config.json`` is missing and *config* is
                not provided, or if a Hub download fails.
            ValueError: If ``model_task`` resolves to an unknown value.

        Examples::

            # Load published nuisance model from Hub:
            model = AutoCageFusion.from_pretrained("sidxz/cage-fusion-nuisance")

            # Transfer learning: load encoder, reset head for a new task:
            new_config = CageFusionConfig(num_labels=12, model_task="regression",
                                          label_names=[...])
            model = AutoCageFusion.from_pretrained(
                "sidxz/cage-fusion-nuisance",
                config=new_config,
                load_backbone_only=True,
            )
        """
        import logging as _logging
        _logger = _logging.getLogger("cagefusion")

        pretrained_model_name_or_path = _resolve_pretrained_path(pretrained_model_name_or_path)

        if load_backbone_only:
            strict = False
            _logger.info(
                "load_backbone_only=True: encoder weights loaded, "
                "task head randomly initialised."
            )

        if config is None:
            config = CageFusionConfig.from_pretrained(pretrained_model_name_or_path)

        model_cls = _TASK_TO_CLASS.get(config.model_task)
        if model_cls is None:
            raise ValueError(
                f"Unknown model_task '{config.model_task}'. "
                f"Supported: {list(_TASK_TO_CLASS)}"
            )
        return model_cls.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            strict=strict,
            **kwargs,
        )
