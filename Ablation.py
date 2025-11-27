"""
Ablation runner for Bayesian_NN_Moe (Bayesian router + sparse Top-k).

What this script does
---------------------
1) Runs a compact, structured sweep over:
     - router training mode: 'expected' vs 'mc' (ELBO)
     - KL weight beta_kl and annealing
     - routing hyperparameters: top_k, temperature tau, capacity_factor
     - optional warm-up: BASE-style / Bayesian-top-k style pretraining
2) Trains via the model's own train_one_epoch(...) (which already implements
   ELBO, KL annealing and optional aux balance losses).
3) Optionally runs a warm-up phase via train_one_epoch_warmup(...) if the
   model exposes that method and the sweep config sets warmup_epochs > 0.
4) After training, probes routing on a few batches to estimate:
     - load-balance metrics (CV, Gini, overflow, entropy, effective experts)
     - specialization metrics:
          * NMI_top1 : NMI between top-1 expert id and class label
          * NMI_topk_soft : NMI between expert and class using soft Top-k weights
5) Evaluates accuracy, NLL, ECE on the test split.
6) Measures training throughput (images/sec) as a proxy for efficiency.
7) Combines everything into:
     - CSV summary files (raw + scored)
     - JSON of the best configuration
     - Several scatter/line plots:
          * Acc vs CV (balance vs performance)
          * Acc vs NMI_topk_soft (data-structure alignment vs performance)
          * ECE vs beta_kl (per router_mode)
          * Overflow vs capacity_factor
          * Img/s vs CV (efficiency vs balance)
          * NMI_topk_soft vs beta_kl (per router_mode)

Design philosophy
-----------------
- The goal is NOT to force perfectly balanced expert loads, but to see how
  different configurations trade off:
    * predictive performance (acc, NLL, ECE),
    * load balance (CV/Gini/overflow),
    * and structural specialization (NMI_topk_soft) between experts and data.
- ELBO-related knobs (router_mode='mc', beta_kl, etc.) and warm-up are treated
  as levers to improve:
    * training stability / efficiency,
    * calibration,
    * and the alignment between expert routing patterns and label structure.
"""

import os
import sys
import time
import json
import math
import argparse
import random
from typing import Dict, Any, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

# Repo-local modules
import cifar10
import cifar100
from Moe.Bayesian_NN_Moe import Bayesian_NN_Moe


# =========================================================
# Calibration & routing metrics
# =========================================================

@torch.no_grad()
def ece_score(logits: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) with equal-width confidence bins.

    Args:
        logits: (N, C) raw logits.
        y     : (N,) ground-truth labels (LongTensor).
        n_bins: number of confidence bins.

    Returns:
        Scalar ECE in [0,1].
    """
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    acc = (pred == y).float()

    bins = torch.linspace(0.0, 1.0, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)

    for i in range(n_bins):
        # mask of samples whose confidence lies in this bin
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            # bin-wise accuracy / confidence
            acc_bin = acc[m].mean()
            conf_bin = conf[m].mean()
            # bin weight = fraction of samples in this bin
            w_bin = m.float().mean()
            ece += torch.abs(acc_bin - conf_bin) * w_bin

    return float(ece.item())


def gini_from_counts(n: np.ndarray) -> float:
    """
    Gini coefficient for non-negative expert counts.

    Definition:
        G = sum_{i,j} |n_i - n_j| / (2 * E * sum_i n_i)

    Args:
        n: (E,) non-negative expert counts.

    Returns:
        Scalar Gini in [0,1]; 0 means perfectly uniform usage.
    """
    n = np.asarray(n, dtype=np.float64)
    s = n.sum()
    if s <= 0:
        return 0.0
    diffsum = np.abs(n[:, None] - n[None, :]).sum()
    return float(diffsum / (2.0 * len(n) * s))


def specialization_nmi(hard_assign: np.ndarray,
                       labels: np.ndarray,
                       num_experts: int) -> float:
    """
    Lightweight NMI(X;Y) / max(H(X), H(Y)) without sklearn.

    Here:
        X = expert id in {0,...,E-1}
        Y = class label in {0,...,C-1} (only observed labels are used).

    Args:
        hard_assign: (N,) predicted expert ids (top-1).
        labels     : (N,) ground-truth labels.
        num_experts: E.

    Returns:
        Normalized Mutual Information in [0,1] (or NaN if invalid).
    """
    x = np.asarray(hard_assign, dtype=int)
    y = np.asarray(labels, dtype=int)
    if x.size == 0:
        return np.nan

    E = num_experts
    classes = np.unique(y)
    eps = 1e-12

    # P(X)
    Px = np.array([(x == e).mean() for e in range(E)], dtype=np.float64) + eps
    # P(Y)
    Py = np.array([(y == c).mean() for c in classes], dtype=np.float64) + eps

    Hx = float(-(Px * np.log(Px)).sum())
    Hy = float(-(Py * np.log(Py)).sum())

    # I(X;Y)
    I = 0.0
    for i, e in enumerate(range(E)):
        for j, c in enumerate(classes):
            Pxy = float(((x == e) & (y == c)).mean()) + eps
            I += Pxy * (math.log(Pxy) - math.log(Px[i]) - math.log(Py[j]))

    return float(I / max(Hx, Hy, eps))


# =========================================================
# Data loader / model builder
# =========================================================

def get_loaders(dataset: str = "cifar100",
                bs: int = 64,
                num_workers: int = 4,
                data_dir: str = "./data") -> Tuple[DataLoader, DataLoader]:
    """
    Construct train/test dataloaders using your repo helpers.

    Args:
        dataset    : 'cifar10' or 'cifar100'.
        bs         : batch size.
        num_workers: DataLoader workers.
        data_dir   : root directory for datasets.

    Returns:
        (train_loader, test_loader)
    """
    if dataset.lower() == "cifar10":
        train, test, _ = cifar10.get_dataloaders(
            setting='2', batch_size=bs, num_workers=num_workers,
            data_dir=data_dir, download=False
        )
    else:
        train, test, _ = cifar100.get_dataloaders(
            setting='2', batch_size=bs, num_workers=num_workers,
            data_dir=data_dir, download=False
        )
    return train, test


def build_model(cfg: Dict[str, Any],
                output_size: int,
                device: str | torch.device) -> torch.nn.Module:
    """
    Instantiate Bayesian_NN_Moe with constructor-aligned argument names.

    Args:
        cfg         : configuration dictionary with model hyperparameters.
        output_size : number of classes for the dataset (10 or 100).
        device      : 'cuda' or 'cpu' (or torch.device).

    Returns:
        A Bayesian_NN_Moe moved to the specified device.
    """
    model = Bayesian_NN_Moe(
        # sizes
        num_experts=cfg["num_experts"],
        num_features=cfg["num_features"],
        output_size=output_size,
        top_k=cfg["top_k"],

        # Bayesian router
        router_prior_var=cfg["prior_var"],
        beta_kl=cfg["beta_kl"],
        kl_anneal_steps=cfg["kl_anneal"],
        router_temperature=cfg["tau"],

        # capacity control
        capacity_factor=cfg["capacity"],          # capacity_factor in the model
        overflow_strategy="drop",

        # router training mode
        router_mode=cfg["router_mode"],           # 'expected' or 'mc'
        elbo_samples=cfg["elbo_samples"],
        use_control_variate=cfg["use_control_variate"],

        # backbone / experts
        backbone_structure=cfg["backbone"],
        backbone_pretrained=False,
        hidden_size=cfg["hidden_size"],

        # optional small balance losses
        w_importance=cfg["w_importance"],
        w_load=cfg["w_load"],
    ).to(device)
    return model


# =========================================================
# Routing probe (post-training)
# =========================================================

@torch.no_grad()
def probe_routing_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    num_experts: int,
    num_classes: int,
    max_batches: int = 40,
) -> Dict[str, float]:
    """
    Probe routing behaviour on a few batches of the *train* loader.

    The goal is to quantify:
      - load balance across experts (CV, Gini, overflow, entropy, effective experts),
      - and how well expert routing aligns with label structure
        (NMI based on top-1 expert id & soft Top-k weights).

    We assume that model(x, return_aux=True) returns:
      logits: (B, C)
      aux   : dict with keys:
               - 'per_expert_counts' : LongTensor[E]      (tokens kept per expert)
               - 'overflow_dropped'  : LongTensor[E]      (tokens dropped due to capacity)
               - 'routing_entropy'   : scalar tensor
               - 'topk_idx'          : LongTensor[B, k]   (expert ids for Top-k)
               - 'topk_weights'      : FloatTensor[B, k]  (renormalized Top-k probs)
    """
    model.eval()
    E = num_experts

    # Accumulate counts across probed batches
    counts = torch.zeros(E, dtype=torch.float64)
    dropped = torch.zeros(E, dtype=torch.float64)
    entropies: List[float] = []

    # For top-1 NMI (hard specialization)
    hard_assign: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    # For soft Top-k NMI: joint weight over (expert e, class c)
    joint_soft = np.zeros((E, num_classes), dtype=np.float64)

    for bidx, (xb, yb) in enumerate(loader):
        if bidx >= max_batches:
            break

        xb = xb.to(device, non_blocking=True)
        logits, aux = model(xb, return_aux=True)

        # 1) Per-expert counts and overflow
        pec = aux.get("per_expert_counts", None)
        ofd = aux.get("overflow_dropped", None)
        if pec is not None:
            counts += pec.detach().cpu().to(torch.float64)
        if ofd is not None:
            dropped += ofd.detach().cpu().to(torch.float64)

        # 2) Routing entropy
        if "routing_entropy" in aux:
            entropies.append(float(aux["routing_entropy"].detach().cpu().item()))

        # 3) Top-1 assignment for NMI_top1
        tki = aux.get("topk_idx", None)  # shape [B, k]
        if tki is not None:
            ha = tki[:, 0].detach().cpu().numpy()
            hard_assign.append(ha)
            labels.append(yb.numpy())

        # 4) Soft Top-k joint distribution P(E,Y) using topk_weights
        tw = aux.get("topk_weights", None)  # shape [B, k]
        if tki is not None and tw is not None:
            tki_np = tki.detach().cpu().numpy()   # (B, k), expert ids
            tw_np = tw.detach().cpu().numpy()     # (B, k), mixing weights
            y_np = yb.numpy().astype(int)         # (B,)

            B, K = tki_np.shape
            # For clarity (not highly optimized): loop over batch and Top-k
            for b in range(B):
                c_id = int(y_np[b])
                if not (0 <= c_id < num_classes):
                    continue
                for j in range(K):
                    e_id = int(tki_np[b, j])
                    w_ij = float(tw_np[b, j])
                    if 0 <= e_id < E:
                        joint_soft[e_id, c_id] += w_ij

    # ---------- Aggregate balance metrics ----------
    counts_np = counts.numpy()
    total_kept = counts_np.sum()
    total_dropped = dropped.numpy().sum()
    total_attempt = total_kept + total_dropped

    if total_kept > 0:
        cv = float(counts_np.std() / (counts_np.mean() + 1e-8))
        gini = gini_from_counts(counts_np)
        eeu = float((counts_np > 0).mean())   # fraction of experts that got at least one token
    else:
        cv = np.nan
        gini = np.nan
        eeu = np.nan

    overflow_rate = float(total_dropped / total_attempt) if total_attempt > 0 else np.nan
    routing_entropy = float(np.mean(entropies)) if len(entropies) > 0 else np.nan

    # ---------- NMI based on top-1 expert id ----------
    nmi_top1 = np.nan
    if len(hard_assign) > 0 and len(labels) > 0:
        ha_all = np.concatenate(hard_assign, axis=0)
        lb_all = np.concatenate(labels, axis=0)
        nmi_top1 = specialization_nmi(ha_all, lb_all, E)

    # ---------- NMI_soft based on soft Top-k joint ----------
    nmi_soft = np.nan
    total_w = float(joint_soft.sum())
    if total_w > 0.0:
        P_ec = joint_soft / total_w         # (E, C) joint distribution
        P_e = P_ec.sum(axis=1)              # (E,)
        P_c = P_ec.sum(axis=0)              # (C,)
        eps = 1e-12

        H_e = float(-np.sum(P_e * np.log(P_e + eps)))
        H_c = float(-np.sum(P_c * np.log(P_c + eps)))
        denom = max(H_e, H_c, eps)

        if denom > 0.0:
            I = 0.0
            for e in range(num_experts):
                for c in range(num_classes):
                    p_ec = P_ec[e, c]
                    if p_ec <= 0.0:
                        continue
                    I += p_ec * (
                        math.log(p_ec + eps)
                        - math.log(P_e[e] + eps)
                        - math.log(P_c[c] + eps)
                    )
            nmi_soft = float(I / denom)

    return dict(
        cv=cv,
        gini=gini,
        overflow=overflow_rate,
        routing_entropy=routing_entropy,
        eeu=eeu,
        # Backward-compatible key: top-1 NMI
        nmi=nmi_top1,
        nmi_top1=nmi_top1,
        nmi_topk_soft=nmi_soft,
    )


# =========================================================
# Evaluation (test set)
# =========================================================

@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: DataLoader,
             device: str | torch.device) -> Dict[str, float]:
    """
    Deterministic evaluation using model(x, return_aux=False).

    Args:
        model : trained Bayesian_NN_Moe model.
        loader: test DataLoader.
        device: 'cuda' or 'cpu'.

    Returns:
        dict with keys: acc, nll, ece
    """
    model.eval()
    logits_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb, return_aux=False)
        logits_all.append(logits.detach().cpu())
        y_all.append(yb.detach().cpu())

    logits = torch.cat(logits_all, dim=0)
    y = torch.cat(y_all, dim=0)

    acc = (logits.argmax(dim=1) == y).float().mean().item()
    nll = F.cross_entropy(logits, y).item()
    ece = ece_score(logits, y)
    return dict(acc=acc, nll=nll, ece=ece)


# =========================================================
# Multi-objective score (for auto hyperparameter selection)
# =========================================================

def score_row(r: Dict[str, Any]) -> float:
    """
    Composite score to compare configurations.

    Intuition:
      - Higher is better for: accuracy, throughput, specialization (NMI_soft).
      - Lower is better for: NLL, ECE, imbalance (CV/Gini), overflow.

    We keep the weights relatively small for balance/specialization terms
    because accuracy & calibration should dominate.
    """
    s = 0.0
    # performance
    s += 2.0 * r.get("acc", 0.0)                  # main objective
    s += 0.5 * r.get("img_s_norm", 0.0)           # normalized throughput
    s -= 0.5 * r.get("nll", 2.0)
    s -= 1.0 * r.get("ece", 0.1) * 10.0           # rescale ECE to ~[0,2]

    # balance
    s -= 0.5 * r.get("cv", 0.5)
    s -= 0.25 * r.get("gini", 0.3)
    s -= 0.5 * r.get("overflow", 0.0) * 100.0     # overflow is usually tiny

    # specialization (soft NMI from Top-k)
    s += 0.3 * r.get("nmi_topk_soft", 0.0)

    return float(s)


# =========================================================
# Default config & sweeps
# =========================================================

def default_base(output_size: int) -> Dict[str, Any]:
    """
    Reasonable starting point for Bayesian_NN_Moe on CIFAR-10/100.

    This base config can be overridden by:
      - CLI arguments (dataset, epochs, batch size, seeds, ...)
      - sweep_list() deltas (router_mode, beta_kl, warmup_epochs, etc.)
    """
    return dict(
        # data & training loop
        dataset="cifar100",
        batch_size=64,
        num_workers=4,
        epochs=40,
        seeds=[0],

        # model / backbone sizes
        backbone="resnet18",
        hidden_size=64,
        num_experts=16,
        num_features=32,

        # routing & Bayesian regularization
        top_k=2,
        tau=1.2,                     # router temperature
        capacity=1.0,                # capacity_factor
        router_mode="expected",      # 'expected' or 'mc'
        elbo_samples=2,              # used when router_mode=='mc'
        use_control_variate=True,
        prior_var=10.0,
        beta_kl=1e-6,
        kl_anneal=5000,

        # optimizer & optional aux balances
        weight_decay=0.02,
        w_importance=0.0,
        w_load=0.0,

        # warm-up configuration
        use_warmup=False,
        warmup_epochs=0,             # set >0 in sweeps to enable warm-up

        # outputs
        output_size=output_size,
    )


def sweep_list() -> List[Dict[str, Any]]:
    """
    Phase-I ablation sweep (compact but informative).

    Each dict here specifies a delta from the base config. We intentionally
    touch only a few key hyperparameters to answer:

      1) What is the impact of ELBO (router_mode='mc', beta_kl, etc.)?
      2) How much does warm-up help (in terms of acc/efficiency/balance/NMI)?
      3) How do routing hyperparameters (top_k, capacity, tau) affect balance
         vs specialization?

    You can expand/modify this list for more exhaustive experiments.
    """
    return [
        # 0. Baseline: expected router, small KL, no warm-up
        {"name": "baseline"},

        # 1. MC-ELBO router, more samples, with control variate
        {"name": "mc_elbo_cv",
         "router_mode": "mc", "elbo_samples": 4, "use_control_variate": True},

        # 2. MC-ELBO router, no control variate (to see variance impact)
        {"name": "mc_elbo_no_cv",
         "router_mode": "mc", "elbo_samples": 4, "use_control_variate": False},

        # 3. No KL (pure maximum likelihood on routing)
        {"name": "no_kl", "beta_kl": 0.0},

        # 4. Slightly stronger KL
        {"name": "kl_stronger", "beta_kl": 1e-5},

        # 5. Lower temperature (sharper gate)
        {"name": "tau_low", "tau": 0.7},

        # 6. Higher temperature (softer gate)
        {"name": "tau_high", "tau": 2.0},

        # 7. Top-1 routing (sparser selection)
        {"name": "top1", "top_k": 1},

        # 8. Top-4 routing (denser selection)
        {"name": "top4", "top_k": 4},

        # 9. Low capacity (more overflow pressure)
        {"name": "capacity_low", "capacity": 0.5},

        # 10. High capacity (less overflow)
        {"name": "capacity_high", "capacity": 2.0},

        # 11. No weight decay (check its influence)
        {"name": "no_wd", "weight_decay": 0.0},

        # 12. Small aux balance losses (importance + load)
        {"name": "aux_balance_small",
         "w_importance": 0.01, "w_load": 0.01},

        # 13. Warm-up only (expected router) before Bayesian training
        {"name": "warmup_expected",
         "use_warmup": True, "warmup_epochs": 10, "router_mode": "expected"},

        # 14. Warm-up + MC-ELBO router
        {"name": "warmup_mc_elbo",
         "use_warmup": True, "warmup_epochs": 10,
         "router_mode": "mc", "elbo_samples": 4, "use_control_variate": True},
    ]


# =========================================================
# Plot helpers
# =========================================================

def plot_scatter(df: pd.DataFrame,
                 x: str,
                 y: str,
                 fname: str,
                 xlabel: Optional[str] = None,
                 ylabel: Optional[str] = None):
    """
    Simple scatter plot helper.

    Args:
        df    : pandas DataFrame containing columns x and y.
        x, y  : column names to plot.
        fname : output filename (e.g., 'plot_acc_vs_cv.png').
        xlabel, ylabel: axis labels (if None, use column names).
    """
    plt.figure()
    plt.scatter(df[x], df[y], s=24)
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_line(df: pd.DataFrame,
              x: str,
              y: str,
              group: Optional[str] = None,
              fname: str = "plot.png",
              xlabel: Optional[str] = None,
              ylabel: Optional[str] = None):
    """
    Simple line plot helper.

    - If group is None: sort by x and draw a single polyline.
    - If group is not None: split by df[group], sort each sub-DF by x, and
      draw one line per group with a legend.

    Args:
        df   : pandas DataFrame.
        x, y : column names.
        group: optional 'hue' column name to group by.
        fname: output filename.
        xlabel, ylabel: axis labels.
    """
    plt.figure()
    if group is None:
        df_sorted = df.sort_values(by=x)
        plt.plot(df_sorted[x], df_sorted[y], marker="o")
    else:
        for g, sub in df.groupby(group):
            sub_sorted = sub.sort_values(by=x)
            plt.plot(sub_sorted[x], sub_sorted[y], marker="o", label=str(g))
        plt.legend()
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


# =========================================================
# Main experiment runner
# =========================================================

def run(args: argparse.Namespace):
    """
    Main ablation entry point.

    Steps:
      1) Build data loaders for CIFAR-10/100.
      2) Construct a base config, override with CLI args.
      3) For each sweep delta and each random seed:
            a) Build a model and optimizer.
            b) Optionally run warm-up (if use_warmup & model exposes warm-up).
            c) Train for cfg['epochs'] epochs via train_one_epoch(...).
            d) Evaluate on test set.
            e) Probe routing for balance & specialization metrics.
            f) Record images/sec throughput.
      4) Normalize throughput, compute composite scores, pick the best row.
      5) Save CSVs, JSON, and generate several diagnostic plots.
    """
    # ---------------- Device and GPU usage ----------------
    # We decide whether to use CUDA based on:
    #   - torch.cuda.is_available()
    #   - args.num_gpus: if 0, force CPU even if CUDA is available.
    use_cuda = torch.cuda.is_available() and (args.num_gpus > 0)
    device = "cuda" if use_cuda else "cpu"

    # world_size controls how we partition the sweep_list across processes.
    # It is typically the total number of GPUs (i.e., processes), but we
    # fall back to 1 when num_gpus == 0 so that all sweeps run on CPU.
    world_size = args.num_gpus if args.num_gpus > 0 else 1

    print(
        f"[INFO] device={device}, "
        f"num_gpus_arg={args.num_gpus}, world_size={world_size}, "
        f"sweep_offset={args.sweep_offset}",
        flush=True,
    )

    # -----------------------------------------------------
    # 1. Dataset & loaders
    # -----------------------------------------------------
    if args.dataset.lower() == "cifar10":
        output_size = 10
        train_loader, test_loader = get_loaders(
            "cifar10", bs=args.bs, num_workers=args.workers, data_dir=args.datadir
        )
    else:
        output_size = 100
        train_loader, test_loader = get_loaders(
            "cifar100", bs=args.bs, num_workers=args.workers, data_dir=args.datadir
        )

    # -----------------------------------------------------
    # 2. Base config + CLI overrides
    # -----------------------------------------------------
    base = default_base(output_size)
    base["dataset"] = args.dataset
    base["batch_size"] = args.bs
    base["num_workers"] = args.workers
    base["epochs"] = args.epochs
    base["seeds"] = [int(s) for s in args.seeds.split(",")]
    base["output_size"] = output_size

    sweeps = sweep_list()
    results: List[Dict[str, Any]] = []

    # -----------------------------------------------------
    # 3. Iterate over sweep variants & seeds
    # -----------------------------------------------------
    for i, delta in enumerate(sweeps):
        # Coarse-grained multi-GPU parallelism:
        # If world_size > 1, each process (identified by sweep_offset)
        # is responsible only for sweeps with index i such that:
        #   i % world_size == args.sweep_offset.
        # This allows you to launch one process per GPU with different
        # sweep_offset values.
        if world_size > 1 and (i % world_size != args.sweep_offset):
            continue

        # Merge base config with this sweep's delta
        cfg = {**base, **delta}
        sweep_name = cfg.get("name", f"sweep_{i}")

        for seed in cfg["seeds"]:
            # 3.a Reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # 3.b Build model & optimizer
            model = build_model(cfg, output_size, device)
            opt = torch.optim.AdamW(
                model.parameters(),
                lr=5e-4,
                weight_decay=cfg["weight_decay"],
            )

            per_epoch_ips: List[float] = []

            # 3.c Optional warm-up phase (BASE-style / Bayesian warm-up)
            #     This uses model.train_one_epoch_warmup(...) if available,
            #     and ONLY optimizes CE loss (no KL, no aux balances).
            warmup_epochs = int(cfg.get("warmup_epochs", 0))
            use_warmup = bool(cfg.get("use_warmup", False)) and warmup_epochs > 0
            if use_warmup and hasattr(model, "train_one_epoch_warmup"):
                print(f"[sweep={i} '{sweep_name}' seed={seed}] "
                      f"Warm-up for {warmup_epochs} epochs...", flush=True)
                for ep in range(warmup_epochs):
                    # Synchronize CUDA before and after timing if we are on GPU
                    if use_cuda:
                        torch.cuda.synchronize()
                    t0 = time.time()

                    avg_loss_w, tr_acc_w = model.train_one_epoch_warmup(
                        loader=train_loader,
                        optimizer=opt,
                        device=device,
                        criterion=None,
                        max_batches=None,   # use full epoch for warm-up
                    )

                    if use_cuda:
                        torch.cuda.synchronize()
                    t1 = time.time()

                    num_imgs = len(train_loader.dataset)
                    ips = num_imgs / max(t1 - t0, 1e-9)
                    per_epoch_ips.append(ips)

                    print(
                        f"  [warmup ep={ep+1}/{warmup_epochs}] "
                        f"loss={avg_loss_w:.4f} acc={tr_acc_w:.4f} ips={ips:.1f}",
                        flush=True,
                    )

            # 3.d Main training loop (Bayesian / ELBO phase)
            print(f"[sweep={i} '{sweep_name}' seed={seed}] "
                  f"Main training for {cfg['epochs']} epochs...", flush=True)

            EPOCHS = max(cfg["epochs"] - warmup_epochs, 1)
            for ep in range(EPOCHS):
                if use_cuda:
                    torch.cuda.synchronize()
                t0 = time.time()

                avg_loss, tr_acc = model.train_one_epoch(
                    loader=train_loader,
                    optimizer=opt,
                    device=device,
                    criterion=None,         # model uses CE internally if None
                    max_batches=None,       # full pass over train set
                )

                if use_cuda:
                    torch.cuda.synchronize()
                t1 = time.time()

                num_imgs = len(train_loader.dataset)
                ips = num_imgs / max(t1 - t0, 1e-9)
                per_epoch_ips.append(ips)

                print(
                    f"  [train ep={ep+1}/{EPOCHS}] "
                    f"loss={avg_loss:.4f} acc={tr_acc:.4f} ips={ips:.1f}",
                    flush=True,
                )

            # 3.e Evaluation on test set
            ev = evaluate(model, test_loader, device)

            # 3.f Routing probe (load-balance + specialization)
            bal = probe_routing_metrics(
                model=model,
                loader=train_loader,
                device=device,
                num_experts=cfg["num_experts"],
                num_classes=output_size,
                max_batches=40,  # small probe; adjust if needed
            )

            # 3.g Throughput summary (median over all epochs, including warm-up)
            img_s = float(np.median(per_epoch_ips)) if len(per_epoch_ips) else np.nan

            # Collect everything into one row
            row = dict(
                sweep_id=i,
                sweep_name=sweep_name,
                seed=seed,
                img_s=img_s,
                use_warmup=use_warmup,
                warmup_epochs=warmup_epochs,
                # performance metrics
                **ev,
                # routing / balance / specialization metrics
                **bal,
                # config fields of interest (for later analysis)
                router_mode=cfg["router_mode"],
                elbo_samples=cfg["elbo_samples"],
                use_control_variate=cfg["use_control_variate"],
                beta_kl=cfg["beta_kl"],
                kl_anneal=cfg["kl_anneal"],
                tau=cfg["tau"],
                top_k=cfg["top_k"],
                capacity=cfg["capacity"],
                weight_decay=cfg["weight_decay"],
                w_importance=cfg["w_importance"],
                w_load=cfg["w_load"],
            )
            results.append(row)

            print(
                f"[sweep={i} '{sweep_name}' seed={seed}] "
                f"acc={row['acc']:.4f} nll={row['nll']:.3f} ece={row['ece']:.4f} "
                f"cv={row.get('cv', np.nan):.3f} gini={row.get('gini', np.nan):.3f} "
                f"overflow={row.get('overflow', np.nan):.4f} "
                f"nmi_topk_soft={row.get('nmi_topk_soft', np.nan):.3f} "
                f"img/s={img_s:.1f}",
                flush=True,
            )

    # -----------------------------------------------------
    # 4. Persist & post-process
    # -----------------------------------------------------
    df = pd.DataFrame(results)
    df.to_csv("ablation_results_raw.csv", index=False)

    # Normalize img/s for scoring
    if df["img_s"].notna().any():
        mn, mx = df["img_s"].min(), df["img_s"].max()
        df["img_s_norm"] = (df["img_s"] - mn) / max(mx - mn, 1e-9)
    else:
        df["img_s_norm"] = 0.0

    # Composite score & best config
    df["score"] = df.apply(score_row, axis=1)
    best_idx = int(df["score"].idxmax())
    best_row = df.iloc[best_idx].to_dict()

    with open("best_config.json", "w") as f:
        json.dump({
            k: best_row.get(k, None)
            for k in [
                "sweep_id", "sweep_name", "seed",
                "router_mode", "elbo_samples", "use_control_variate",
                "beta_kl", "kl_anneal", "tau", "top_k", "capacity",
                "weight_decay", "w_importance", "w_load",
                "use_warmup", "warmup_epochs",
                "score", "acc", "ece", "nll",
                "cv", "gini", "overflow",
                "nmi_top1", "nmi_topk_soft",
                "img_s",
            ]
        }, f, indent=2)

    df.to_csv("ablation_results_scored.csv", index=False)

    # -----------------------------------------------------
    # 5. Diagnostic plots
    # -----------------------------------------------------

    # (1) Accuracy vs CV (balance vs performance)
    if "cv" in df.columns:
        plot_scatter(
            df.dropna(subset=["cv", "acc"]),
            x="cv",
            y="acc",
            fname="plot_acc_vs_cv.png",
            xlabel="CV of expert loads (lower = more balanced)",
            ylabel="Accuracy",
        )

    # (2) Accuracy vs NMI_topk_soft (specialization vs performance)
    if "nmi_topk_soft" in df.columns:
        plot_scatter(
            df.dropna(subset=["nmi_topk_soft", "acc"]),
            x="nmi_topk_soft",
            y="acc",
            fname="plot_acc_vs_nmi_soft.png",
            xlabel="NMI_topk_soft (expert–label alignment)",
            ylabel="Accuracy",
        )

    # (3) ECE vs beta_kl (grouped by router_mode)
    if "beta_kl" in df.columns and "router_mode" in df.columns:
        plot_line(
            df,
            x="beta_kl",
            y="ece",
            group="router_mode",
            fname="plot_ece_vs_beta.png",
            xlabel="beta_kl (router KL weight)",
            ylabel="ECE",
        )

    # (4) Overflow vs capacity_factor
    if "capacity" in df.columns and "overflow" in df.columns:
        plot_line(
            df,
            x="capacity",
            y="overflow",
            group=None,
            fname="plot_overflow_vs_capacity.png",
            xlabel="Capacity factor",
            ylabel="Overflow rate",
        )

    # (5) Images/s vs CV (efficiency–balance relationship)
    if "img_s" in df.columns and "cv" in df.columns:
        plot_scatter(
            df.dropna(subset=["cv", "img_s"]),
            x="cv",
            y="img_s",
            fname="plot_imgs_vs_cv.png",
            xlabel="CV of expert loads",
            ylabel="Images/sec (throughput)",
        )

    # (6) NMI_topk_soft vs beta_kl (grouped by router_mode)
    if "nmi_topk_soft" in df.columns and "beta_kl" in df.columns:
        plot_line(
            df.dropna(subset=["nmi_topk_soft", "beta_kl"]),
            x="beta_kl",
            y="nmi_topk_soft",
            group="router_mode",
            fname="plot_nmi_soft_vs_beta.png",
            xlabel="beta_kl (router KL weight)",
            ylabel="NMI_topk_soft",
        )

    print("\nSaved files:")
    print("  - ablation_results_raw.csv")
    print("  - ablation_results_scored.csv")
    print("  - best_config.json")
    print("  - plots:")
    print("      * plot_acc_vs_cv.png")
    print("      * plot_acc_vs_nmi_soft.png")
    print("      * plot_ece_vs_beta.png")
    print("      * plot_overflow_vs_capacity.png")
    print("      * plot_imgs_vs_cv.png")
    print("      * plot_nmi_soft_vs_beta.png")


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablations for Bayesian_NN_Moe")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar10", "cifar100"],
        help="Dataset choice.",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="./data",
        help="Data directory.",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=128,
        help="Batch size.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of main training epochs per sweep variant "
             "(warm-up epochs are configured inside sweep_list).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated seeds, e.g. '0,1,2'.",
    )

    # ---------- New CLI arguments for GPU / sweep parallelism ----------
    # num_gpus:
    #   - default is 1 if CUDA is available, otherwise 0.
    #   - interpreted as the total number of GPUs (i.e., concurrent processes)
    #     used for coarse-grained sweep parallelism.
    default_num_gpus = 1 if torch.cuda.is_available() else 0
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=default_num_gpus,
        help=(
            "Total number of GPUs used for coarse-grained sweep parallelism. "
            "If 0, the script runs on CPU only. "
            "Typically you launch one process per GPU with the same --num-gpus."
        ),
    )

    # sweep_offset:
    #   - which remainder class this process is responsible for when partitioning
    #     sweep_list by index modulo num_gpus.
    #   - for example, with --num-gpus=2, you can launch two processes with
    #     --sweep-offset=0 and --sweep-offset=1, each running half of the sweeps.
    parser.add_argument(
        "--sweep-offset",
        type=int,
        default=0,
        help=(
            "Index of this process in [0, num_gpus-1] for partitioning the "
            "sweep_list. Only sweeps with index % num_gpus == sweep_offset "
            "are executed by this process."
        ),
    )

    args = parser.parse_args()
    run(args)
