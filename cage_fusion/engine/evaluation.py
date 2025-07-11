import os
import torch
import numpy as np
import random
from tqdm import tqdm
from cage_fusion.utils.logging_utils import logger
from .metrics import (
    AUCBatchAggregatorToDisk,
    MCCBatchAggregatorToDisk,
    PRBatchAggregatorToDisk,
)
from .utils import (
    move_bmg_to_device,
)

from cage_fusion.viz.token_viz import (
    visualize_top_token_attentions,
    visualize_attention_weights,
    visualize_total_atom_contribution,
    visualize_combined_atom_contribution
)


from cage_fusion.viz.prompt_viz import visualize_fg_attention

# ----------------------------------------------------


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    criterion,
    device,
    num_tasks,
    label_names,
    threshold_search=np.linspace(0.1, 0.9, 20),
    use_precomputed_thresholds=None,
    plot_attn=False,
    cache_dir="val_cache",
    tokenizer_obj=None,
):
    """
    Evaluates the model using batched metrics aggregation with minimal memory overhead.
    """
    logger.info(f"Evaluating model with cache directory: {cache_dir}")

    model.eval()
    total_loss = 0.0

    # Select a random batch for attention visualization if requested
    batch_to_log_idx = -1
    if plot_attn and len(loader) > 0:
        batch_to_log_idx = random.randint(0, len(loader) - 1)
        logger.info(
            f"Evaluation attention will be logged for batch index: {batch_to_log_idx}"
        )

    total_graph_norm, total_attn_norm, total_aux_norm = 0.0, 0.0, 0.0

    mcc_agg = MCCBatchAggregatorToDisk(
        num_tasks, os.path.join(cache_dir, "mcc"), label_names
    )
    auc_agg = AUCBatchAggregatorToDisk(
        num_tasks, os.path.join(cache_dir, "auc"), label_names
    )
    pr_agg = PRBatchAggregatorToDisk(
        num_tasks, os.path.join(cache_dir, "pr"), label_names
    )

    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        if batch is None:
            continue

        # Unpack all 7 items from the batch, including SMILES
        (
            bmg,
            token_embs,
            attn_mask,
            rdkit_feats,
            labels,
            input_ids_batch,
            smiles_batch,
        ) = batch

        bmg = move_bmg_to_device(bmg, device)
        token_embs, attn_mask = token_embs.to(device), attn_mask.to(device)
        rdkit_feats, labels = rdkit_feats.to(device), labels.to(device)
        input_ids_batch = input_ids_batch.to(device)

        # Only request attention weights for the randomly selected batch
        should_return_attn = batch_idx == batch_to_log_idx

        outputs = model(
            bmg=bmg,
            sequence_embeddings=token_embs,
            attn_mask=attn_mask,
            aux_feats=rdkit_feats,
            input_ids_batch=input_ids_batch,
            smiles_batch=smiles_batch,
            return_attn=should_return_attn,
        )

        # Unpack the 7-item model output
        (
            logits,
            _,
            _,
            g2t_weights,
            t2a_weights,
            attn_output,
            graph_repr,
            _,
            prompt_attn_weights,
        ) = outputs

        if should_return_attn:
            # Select a random sample from the chosen batch for visualization
            random_sample_idx = random.randint(0, len(smiles_batch) - 1)
            # logger.info(
            #     f"Visualizing random sample index {random_sample_idx} from batch {batch_idx}."
            # )

            plot_dir = os.path.join(cache_dir, "attention_plots_eval")
            os.makedirs(plot_dir, exist_ok=True)

            # Plot 1: Standard Graph-to-Token Attention Heatmap for the random sample
            if g2t_weights is not None:
                plot_path_g2t = os.path.join(plot_dir, "g2t_evaluation_attention.png")
                visualize_attention_weights(
                    g2t_weights[random_sample_idx],
                    attn_mask[random_sample_idx],
                    model.num_heads,
                    output_path=plot_path_g2t,
                    input_ids=input_ids_batch[random_sample_idx],
                    tokenizer_obj=tokenizer_obj,
                    smiles=smiles_batch[random_sample_idx],
                )

            # Plot 2: Visualize atom attention from the TOP-attended tokens
            if g2t_weights is not None and t2a_weights is not None:
                smiles_to_plot = smiles_batch[random_sample_idx]
                if smiles_to_plot:
                    # Step 1: Find the most important tokens for the chosen sample
                    token_scores = g2t_weights[random_sample_idx].sum(dim=(0, 1))

                    special_ids = [
                        tokenizer_obj.pad_token_id,
                        tokenizer_obj.cls_token_id,
                        tokenizer_obj.sep_token_id,
                    ]
                    for special_id in special_ids:
                        token_scores[
                            input_ids_batch[random_sample_idx] == special_id
                        ] = -1e9

                    num_top_tokens_to_plot = 3
                    top_token_indices = torch.argsort(token_scores, descending=True)[
                        :num_top_tokens_to_plot
                    ]
                    # Get the full list of tokens for the molecule
                    full_token_list_ids = (
                        input_ids_batch[random_sample_idx].cpu().numpy()
                    )

                    # Step 2: Bundle the top token info together
                    # top_tokens_info = []
                    # for token_idx_tensor in top_token_indices:
                    #     token_idx = token_idx_tensor.item()
                    #     token_str = tokenizer_obj.convert_ids_to_tokens(
                    #         [input_ids_batch[random_sample_idx][token_idx].item()]
                    #     )[0]
                    #     top_tokens_info.append((token_idx, token_str))
                    actual_tokens_mask = (
                        full_token_list_ids != tokenizer_obj.pad_token_id
                    )
                    actual_tokens_ids = full_token_list_ids[actual_tokens_mask]
                    full_token_list_str = tokenizer_obj.convert_ids_to_tokens(
                        actual_tokens_ids
                    )

                    # Step 3: Make a single call to the new visualization function
                    visualize_top_token_attentions(
                        smiles=smiles_to_plot,
                        attention_weights=t2a_weights[random_sample_idx],
                        full_token_list=full_token_list_str,
                        # top_tokens_info=top_tokens_info,
                        top_token_indices=top_token_indices.cpu().numpy(),
                        output_dir=plot_dir,
                    )
                    
                    # Step 4 Total attention contribution
                    attn = t2a_weights[random_sample_idx]
                    if attn.ndim == 3:  # [n_heads, n_tokens, n_atoms]
                        attn = attn.mean(axis=0)
                    # Get logit or predicted value for this sample
                    logit_vec = logits[random_sample_idx].detach().cpu().numpy()
                    task_idx = np.argmax(np.abs(logit_vec))  # abs() finds the most extreme value, or remove abs() for just highest positive
                    pred_logit = logit_vec[task_idx]  # this will be a scalar
                    visualize_total_atom_contribution(
                        smiles=smiles_to_plot,
                        t2a_weights_sample=attn,
                        pred_logit=pred_logit,
                        output_path=os.path.join(plot_dir, "atom_total_contrib.png")
                    )
                else:
                    logger.warning(
                        "SMILES for the chosen random sample is empty. Skipping T2A visualization."
                    )
            # Plot 3: Visualize functional group attention
            if prompt_attn_weights and prompt_attn_weights[random_sample_idx]:
                plot_path_fg = os.path.join(plot_dir, "fg_prompt_attention.png")
                visualize_fg_attention(
                    smiles=smiles_batch[random_sample_idx],
                    prompt_attn_weights=prompt_attn_weights[random_sample_idx],
                    output_path=plot_path_fg,
                    title=f"Functional Group Attention (PROMPT)",
                )
                
                weight_fg = float(model.alpha.detach().cpu().item())
                weight_t2a = float(model.scale_graph.detach().cpu().item())

                visualize_combined_atom_contribution(
                    smiles=smiles_to_plot,
                    t2a_weights_sample=attn,
                    pred_logit=pred_logit,
                    prompt_attn_weights=prompt_attn_weights[random_sample_idx],
                    output_path=os.path.join(plot_dir, "atom_combined_contrib.png"),
                    weight_t2a=weight_t2a,
                    weight_fg=weight_fg,

                )

        loss = criterion(logits, labels)
        total_loss += loss.item()
        if graph_repr is not None:
            total_graph_norm += graph_repr.norm(dim=1).mean().item()
        if attn_output is not None:
            total_attn_norm += attn_output.norm(dim=1).mean().item()
        if rdkit_feats is not None:
            total_aux_norm += rdkit_feats.norm(dim=1).mean().item()

        probs = torch.sigmoid(logits)
        mcc_agg.update(labels, probs)
        auc_agg.update(labels, probs)
        pr_agg.update(labels, probs)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    per_task_auc = auc_agg.compute(reduce="none")
    per_task_pr = pr_agg.compute(reduce="none")

    # --- FIXED: Use len() for robustness, as it works for both lists and numpy arrays ---
    avg_auc = float(np.nanmean(per_task_auc)) if len(per_task_auc) > 0 else 0.0
    avg_pr = float(np.mean(per_task_pr)) if len(per_task_pr) > 0 else 0.0

    avg_graph_norm = total_graph_norm / len(loader) if len(loader) > 0 else 0
    avg_attn_norm = total_attn_norm / len(loader) if len(loader) > 0 else 0
    avg_aux_norm = total_aux_norm / len(loader) if len(loader) > 0 else 0

    if use_precomputed_thresholds is not None:
        avg_mcc, _, per_task_mcc = mcc_agg.compute(
            thresholds=use_precomputed_thresholds
        )
        best_thresholds = use_precomputed_thresholds
    else:
        avg_mcc, best_thresholds, per_task_mcc = mcc_agg.compute(
            threshold_search=threshold_search
        )

    per_task_metrics = list(zip(per_task_mcc, per_task_auc, per_task_pr))

    return (
        avg_loss,
        avg_mcc,
        avg_auc,
        avg_pr,
        np.array(best_thresholds),
        per_task_metrics,
        avg_graph_norm,
        avg_attn_norm,
        avg_aux_norm,
    )
