# In cage_fusion/viz/token_viz.py

import matplotlib.pyplot as plt
import numpy as np
from cage_fusion.engine.fg_utils import FG_NAMES # Import functional group names

def visualize_fg_attention(prompt_attn_weights, output_path, title=""):
    """
    Visualizes functional group attention weights using a polar plot.

    Args:
        prompt_attn_weights (dict): A dictionary containing 'fg_ids' and 'weights'.
        output_path (str): The path to save the generated plot.
        title (str): The title for the plot.
    """
    fg_ids = prompt_attn_weights.get("fg_ids", [])
    weights = prompt_attn_weights.get("weights", [])

    if not fg_ids:
        print(f"No functional groups provided for visualization.")
        return

    labels = [FG_NAMES[i] for i in fg_ids]
    
    # --- Create Polar Plot ---
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    weights = np.concatenate((weights, [weights[0]])) # Close the plot
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, weights, color='teal', alpha=0.4)
    ax.plot(angles, weights, color='teal', linewidth=2)
    
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    if title:
        plt.title(title, size=14, color='teal', y=1.1)
    
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
