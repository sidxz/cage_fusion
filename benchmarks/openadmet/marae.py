"""
benchmarks/openadmet/marae.py
==============================
Standalone MA-RAE scorer that exactly matches the OpenADMET leaderboard formula.

Use this to verify local scores against the leaderboard before/after submission.

Formula
-------
Per-endpoint::

    RAE_i = MAE_i / mean(|y_true_i - mean(y_true_i)|)

Overall::

    MA-RAE = mean(RAE_i)   over endpoints with >= 2 valid samples

Both ``y_true`` and ``y_pred`` must be in **log scale** (the same scale used
during training).  Do NOT inverse-transform before calling this function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


def compute_marae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
    min_samples: int = 2,
) -> dict:
    """Compute MA-RAE and per-endpoint metrics.

    Args:
        y_true:      (N, T) ground-truth array in log scale.
        y_pred:      (N, T) prediction array in log scale.
        label_names: Names for the T endpoints.
        min_samples: Minimum valid (non-NaN) samples required to include an
                     endpoint in the MA-RAE calculation (default 2).

    Returns:
        Dict with keys:

        ``"ma_rae"``
            float — primary leaderboard metric (lower is better).
        ``"per_endpoint"``
            dict mapping endpoint name → ``{"n", "mae", "rae", "r2",
            "spearman", "kendall"}``.
        ``"num_endpoints_scored"``
            int — number of endpoints included in MA-RAE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

    n_tasks     = y_true.shape[1]
    label_names = label_names or [f"Task {i}" for i in range(n_tasks)]

    raes: list[float] = []
    per_endpoint: dict = {}

    for i, name in enumerate(label_names):
        mask = ~np.isnan(y_true[:, i])
        yt   = y_true[mask, i]
        yp   = y_pred[mask, i]
        n    = int(mask.sum())

        if n < min_samples:
            per_endpoint[name] = {"n": n, "mae": np.nan, "rae": np.nan,
                                  "r2": np.nan, "spearman": np.nan, "kendall": np.nan}
            continue

        mae  = float(mean_absolute_error(yt, yp))
        denom = float(np.mean(np.abs(yt - np.mean(yt))))
        rae  = mae / denom if denom > 1e-12 else np.nan

        r2   = float(r2_score(yt, yp)) if n > 1 else np.nan
        spr  = float(spearmanr(yt, yp).statistic)  if n > 1 else np.nan
        ktau = float(kendalltau(yt, yp).statistic) if n > 1 else np.nan

        if not np.isnan(rae):
            raes.append(rae)

        per_endpoint[name] = {
            "n": n, "mae": mae, "rae": rae,
            "r2": r2, "spearman": spr, "kendall": ktau,
        }

    ma_rae = float(np.mean(raes)) if raes else np.nan
    return {
        "ma_rae":               ma_rae,
        "per_endpoint":         per_endpoint,
        "num_endpoints_scored": len(raes),
    }


def print_report(results: dict, leaderboard_top: float | None = 0.5113) -> None:
    """Pretty-print a MA-RAE evaluation report.

    Args:
        results:          Return value of :func:`compute_marae`.
        leaderboard_top:  Leaderboard #1 MA-RAE for comparison (default 0.5113).
    """
    rows = []
    for name, m in results["per_endpoint"].items():
        rows.append({
            "Endpoint": name,
            "N":        m["n"],
            "MAE":      f"{m['mae']:.4f}"      if not np.isnan(m["mae"])      else "—",
            "RAE":      f"{m['rae']:.4f}"      if not np.isnan(m["rae"])      else "—",
            "R²":       f"{m['r2']:.4f}"       if not np.isnan(m["r2"])       else "—",
            "Spearman": f"{m['spearman']:.4f}" if not np.isnan(m["spearman"]) else "—",
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    print(f"  MA-RAE (ours)              : {results['ma_rae']:.4f}")
    if leaderboard_top is not None:
        delta = results["ma_rae"] - leaderboard_top
        sign  = "+" if delta >= 0 else ""
        print(f"  MA-RAE (leaderboard #1)    : {leaderboard_top:.4f}")
        print(f"  Delta vs leaderboard #1    : {sign}{delta:.4f}  "
              f"({'worse' if delta > 0 else 'better'})")
    print(f"  Endpoints scored           : {results['num_endpoints_scored']}")
