import os
import json
import wandb
import numpy as np
from collections import defaultdict
from pprint import pprint

from . import node_evaluation
from data_utils import *
from provnet_utils import log
from .evaluation_utils import *


def standard_evaluation(cfg, evaluation_fn):
    test_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "test")
    val_losses_dir = os.path.join(cfg.detection.gnn_testing._edge_losses_dir, "val")
    
    tw_to_malicious_nodes = compute_tw_labels(cfg)
    
    best_mcc, best_stats = -1e6, {}
    for model_epoch_dir in listdir_sorted(test_losses_dir):
        log(f"\nEvaluation of model {model_epoch_dir}...")

        test_tw_path = os.path.join(test_losses_dir, model_epoch_dir)
        val_tw_path = os.path.join(val_losses_dir, model_epoch_dir)

        stats = evaluation_fn(val_tw_path, test_tw_path, model_epoch_dir, cfg, tw_to_malicious_nodes=tw_to_malicious_nodes)
            
        out_dir = cfg.detection.evaluation.node_evaluation._precision_recall_dir
        stats["epoch"] = int(model_epoch_dir.split("_")[-1])
        stats["simple_scores_img"] = wandb.Image(os.path.join(out_dir, f"simple_scores_{model_epoch_dir}.png"))
        
        scores = os.path.join(out_dir, f"scores_{model_epoch_dir}.png")
        if os.path.exists(scores):
            stats["scores_img"] = wandb.Image(scores)
        
        dor = os.path.join(out_dir, f"dor_{model_epoch_dir}.png")
        if os.path.exists(dor):
            stats["dor_img"] = wandb.Image(dor)
        
        pr = os.path.join(out_dir, f"pr_curve_{model_epoch_dir}.png")
        if os.path.exists(pr):
            stats["precision_recall_img"] = wandb.Image(pr)
        
        wandb.log(stats)
        
        if stats["mcc"] > best_mcc:
            best_mcc = stats["mcc"]
            best_stats = stats

    wandb.log(best_stats)
    _save_eval_log(cfg, best_stats)


def _save_eval_log(cfg, best_stats):
    gnn = cfg.detection.gnn_training
    drop = gnn.encoder.graph_attention.dropout
    filename = (
        f"{cfg.dataset.name}"
        f"__ep{gnn.num_epochs}_lr{gnn.lr}"
        f"_hid{gnn.node_hid_dim}_out{gnn.node_out_dim}"
        f"_drop{drop}_ratio{cfg._train_ratio}"
        f"_seed{cfg._seed}"
        f".log"
    )

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(project_root, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    stats_lines = "\n".join(
        f"  {k}: {v}" for k, v in best_stats.items() if not hasattr(v, '__module__')
    )
    entry = f"[{cfg._exp_start_time}]\n{stats_lines}\n"

    with open(log_path, "a") as f:
        if os.path.getsize(log_path) > 0:
            f.write("\n----\n")
        f.write(entry)

    log(f"Eval log saved to {log_path}")


def main(cfg):
    method = cfg.detection.evaluation.used_method.strip()
    if method == "node_evaluation":
        standard_evaluation(cfg, evaluation_fn=node_evaluation.main)
    else:
        raise ValueError(f"Invalid evaluation method {cfg.detection.evaluation.used_method}")


if __name__ == "__main__":
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
