"""
Moe_Compare_Models.py

Compare multiple MoE variants on CIFAR-10 / CIFAR-100 with SVHN as OOD.

Models:
  - Bayesian_NN_Moe  (our Bayesian router with ELBO, using best_config.json)
  - Simple_Moe       (Shazeer-style aux-loss MoE)
  - Expert_Choice    (EC MoE)
  - Aux_Free_Moe     (auxiliary-loss-free balancing)
  - BASE_Moe         (balanced assignment MoE)

Datasets:
  - ID:  cifar10, cifar100
  - OOD: svhn (via cifar100.get_ood_dataloader)

Metrics (intentionally minimal but MoE-aware):
  - test_acc, test_nll, ECE
  - routing CV (per_expert_counts), NMI(expert; label), overflow rate
  - ID / OOD max-softmax confidence stats (mean, std)

Bayes-aware extra metrics (computed for all MoE models, but conceptually Bayes-friendly):
  - H_E_norm  : normalized entropy of expert usage
  - H_Y_norm  : normalized entropy of label distribution
  - EAS       : entropy alignment score 1 - |H_E_norm - H_Y_norm|
  - DARS      : Data-aware Routing Score = 0.5 * EAS + 0.5 * NMI

Warm-up (Bayesian only):
  - Optional warm-up epochs before main training.
  - If model exposes train_one_epoch_warmup(...), use it.
  - Otherwise: temporarily disable KL (beta_kl=0), set kl_anneal_steps=0,
    and force router_mode='expected' during warm-up, then restore the
    original settings.

Results are written to a CSV file (moe_compare_results.csv).

Usage examples:
  # Compare all models on CIFAR-100, with SVHN as OOD, and 5 warm-up epochs for Bayes
  python Moe_Compare_Models.py --dataset cifar100 --models bayes,simple,ec,auxfree,base --ood svhn --epochs 40 --bayes-warmup-epochs 5

  # CIFAR-10 only, fewer epochs, no warm-up
  python Moe_Compare_Models.py --dataset cifar10 --models bayes,simple --ood svhn --epochs 20
"""

import os
import sys
import math
import json
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# add repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# project-local helpers
import cifar10
import cifar100

# MoE models (adjust paths if your files are located differently)
from Moe.Bayesian_NN_Moe import Bayesian_NN_Moe
from Moe.Simple_Moe import Simple_Moe
from Moe.Expert_Choice_Moe import Expert_Choice
from Moe.Aux_Free_Moe import Aux_Free_Moe
from Moe.BASE_Moe import BASE_Moe


# ---------------------------------------------------------
# Basic metrics: ECE, NMI, entropy alignment
# ---------------------------------------------------------

@torch.no_grad()
def ece_score(logits: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) with equal-width confidence bins.
    Works for any K-class classification problem.
    """
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    acc = (pred == y).float()

    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += torch.abs(acc[m].mean() - conf[m].mean()) * m.float().mean()
    return float(ece.item())


def specialization_nmi(hard_assign: np.ndarray,
                       labels: np.ndarray,
                       num_experts: int) -> float:
    """
    Normalize I(X;Y) by max(H(X), H(Y)) where:
      X = expert id in [0..E-1]
      Y = class id (any integer labels)

    This measures how much *specialization* experts have w.r.t. labels;
    it is label-semantic-agnostic and works for any labeled dataset.
    """
    x = np.asarray(hard_assign, dtype=int)
    y = np.asarray(labels, dtype=int)
    if x.size == 0:
        return float("nan")

    E = num_experts
    classes = np.unique(y)
    eps = 1e-12

    # P(X)
    Px = np.array([(x == e).mean() for e in range(E)], dtype=np.float64) + eps
    # P(Y)
    Py = np.array([(y == c).mean() for c in classes], dtype=np.float64) + eps

    Hx = float(-(Px * np.log(Px)).sum())
    Hy = float(-(Py * np.log(Py)).sum())

    # mutual information I(X;Y)
    I = 0.0
    for i, e in enumerate(range(E)):
        for j, c in enumerate(classes):
            Pxy = float(((x == e) & (y == c)).mean()) + eps
            I += Pxy * (math.log(Pxy) - math.log(Px[i]) - math.log(Py[j]))

    return float(I / max(Hx, Hy))


def _entropy(p: np.ndarray) -> float:
    """
    Helper: Shannon entropy with natural log.
    Assumes p is a valid probability vector (sums to 1).
    """
    p = np.asarray(p, dtype=np.float64) + 1e-12
    return float(-(p * np.log(p)).sum())


def entropy_alignment_metrics(counts_np: np.ndarray,
                              labels_np: np.ndarray,
                              num_experts: int,
                              num_classes: int) -> Tuple[float, float, float]:
    """
    Compute entropy-based alignment metrics between expert usage and data distribution.

    Returns:
        H_E_norm : normalized expert entropy H(E) / log(num_experts)
        H_Y_norm : normalized label entropy H(Y) / log(num_classes)
        EAS      : entropy alignment score = 1 - |H_E_norm - H_Y_norm|
    """
    # Edge cases: if no counts or no labels, return NaNs
    if counts_np.sum() <= 0 or labels_np.size == 0:
        return float("nan"), float("nan"), float("nan")

    # Expert usage distribution p(E)
    p_e = counts_np.astype(np.float64) / counts_np.sum()
    H_E = _entropy(p_e)
    H_E_norm = H_E / max(math.log(num_experts + 1e-12), 1e-12)

    # Label distribution p(Y)
    label_counts = np.bincount(labels_np.astype(int), minlength=num_classes).astype(np.float64)
    if label_counts.sum() <= 0:
        return H_E_norm, float("nan"), float("nan")
    p_y = label_counts / label_counts.sum()
    H_Y = _entropy(p_y)
    H_Y_norm = H_Y / max(math.log(num_classes + 1e-12), 1e-12)

    # Entropy Alignment Score: 1 - |H_E_norm - H_Y_norm|, clipped to [0,1]
    eas = 1.0 - abs(H_E_norm - H_Y_norm)
    eas = float(max(0.0, min(1.0, eas)))
    return H_E_norm, H_Y_norm, eas


# ---------------------------------------------------------
# Dataset builders (ID + OOD)
# ---------------------------------------------------------

def build_id_loaders(dataset_name: str,
                     batch_size: int,
                     num_workers: int,
                     download: bool = True) -> Tuple[DataLoader, DataLoader, int]:
    """
    Build ID train/test loaders using your existing helpers.
    Returns (train_loader, test_loader, num_classes).
    """
    dataset_name = dataset_name.lower()
    if dataset_name == "cifar10":
        train_loader, test_loader, _ = cifar10.get_dataloaders(
            "2", batch_size=batch_size, num_workers=num_workers,
            data_dir="./data", download=download
        )
        num_classes = 10
    elif dataset_name == "cifar100":
        train_loader, test_loader, _ = cifar100.get_dataloaders(
            "2", batch_size=batch_size, num_workers=num_workers,
            data_dir="./data", download=download
        )
        num_classes = 100
    else:
        raise ValueError(f"Unsupported ID dataset: {dataset_name}")
    return train_loader, test_loader, num_classes


def build_ood_loader(name: str,
                     batch_size: int,
                     num_workers: int,
                     download: bool = True) -> DataLoader:
    """
    Build an OOD dataloader using your CIFAR-100 helpers.

    Supported OOD names:
      - svhn, stl10, cifar10  (via cifar100.get_ood_dataloader)
      - cifar100              (CIFAR-100 test split)
    """
    name = name.lower()
    if name in ("svhn", "stl10", "cifar10"):
        return cifar100.get_ood_dataloader(
            name, batch_size=batch_size, num_workers=num_workers,
            data_dir="./data", download=download
        )
    if name == "cifar100":
        return cifar100.get_dataloaders(
            "2", batch_size=batch_size, num_workers=num_workers,
            data_dir="./data", download=download
        )[1]
    raise ValueError(f"Unsupported OOD dataset: {name}")


# ---------------------------------------------------------
# Load best config for Bayesian_NN_Moe (from ablations)
# ---------------------------------------------------------

def load_best_config(path: str) -> Dict[str, Any]:
    """
    Load best_config.json produced by your ablation script, if present.
    If missing or invalid, fall back to an empty dict.
    """
    if not os.path.isfile(path):
        print(f"[WARN] best_config.json not found at {path}, using default Bayesian config.")
        return {}
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
        print(f"[INFO] Loaded best Bayesian config from {path}:")
        print(cfg)
        return cfg
    except Exception as e:
        print(f"[WARN] Failed to load best_config.json ({e}), using defaults.")
        return {}


# ---------------------------------------------------------
# Model builders for each MoE variant
# ---------------------------------------------------------

def build_bayesian_moe(num_classes: int,
                       device: torch.device,
                       best_cfg: Dict[str, Any]) -> nn.Module:
    """
    Instantiate Bayesian_NN_Moe using a combination of:
      - reasonable defaults (aligned with your ablation script)
      - overrides from best_config.json (if provided)
    """
    # defaults (aligned with default_base in Ablation.py)
    base = dict(
        num_experts=16,
        num_features=32,
        hidden_size=64,
        top_k=2,
        tau=1.2,
        capacity=1.0,
        router_mode="expected",
        elbo_samples=2,
        use_control_variate=True,
        prior_var=10.0,
        beta_kl=1e-6,
        kl_anneal=5000,
        w_importance=0.0,
        w_load=0.0,
    )
    cfg = {**base, **best_cfg}  # best_cfg keys (e.g., tau, beta_kl, etc.) override defaults

    model = Bayesian_NN_Moe(
        # sizes
        num_experts=cfg["num_experts"],
        num_features=cfg["num_features"],
        output_size=num_classes,
        top_k=cfg["top_k"],

        # Bayesian router hyperparameters
        router_prior_var=cfg.get("prior_var", 10.0),
        beta_kl=cfg.get("beta_kl", 1e-6),
        kl_anneal_steps=cfg.get("kl_anneal", 5000),
        router_temperature=cfg.get("tau", 1.2),

        # capacity control
        capacity_factor=cfg.get("capacity", 1.0),
        overflow_strategy="drop",

        # router training mode
        router_mode=cfg.get("router_mode", "expected"),    # 'expected' or 'mc'
        elbo_samples=cfg.get("elbo_samples", 2),
        use_control_variate=cfg.get("use_control_variate", True),

        # backbone / experts
        backbone_structure="resnet18",
        backbone_pretrained=False,
        hidden_size=cfg["hidden_size"],

        # optional balance losses
        w_importance=cfg.get("w_importance", 0.0),
        w_load=cfg.get("w_load", 0.0),
    ).to(device)

    return model


def build_simple_moe(num_classes: int,
                     device: torch.device) -> nn.Module:
    """
    Shazeer-style Sparse MoE with auxiliary losses.
    Parameters aligned with your Simple_Moe quick test.
    """
    model = Simple_Moe(
        num_experts=16,
        top_k=2,
        aux_loss_weight=0.0,                 # we use explicit w_importance / w_load instead
        backbone_structure="resnet18",
        backbone_pretrained=False,
        num_features=32,
        hidden_size=64,
        output_size=num_classes,
        per_token_noise=True,
        min_noise_scale=1e-2,
        w_importance=0.01,
        w_load=0.01,
        capacity_factor=1.5,
        overflow_strategy="drop",
        router_temperature=1.0,
    ).to(device)
    return model


def build_expert_choice(num_classes: int,
                        device: torch.device) -> nn.Module:
    """
    Expert Choice MoE (EC).
    """
    model = Expert_Choice(
        num_experts=16,
        backbone_structure="resnet18",
        backbone_pretrained=False,
        num_features=32,
        hidden_size=64,
        output_size=num_classes,
        capacity_factor=1.0,
        use_noisy_scores=False,
    ).to(device)
    return model


def build_aux_free(num_classes: int,
                   device: torch.device) -> nn.Module:
    """
    Aux-free MoE (DeepSeek-style).
    """
    model = Aux_Free_Moe(
        num_experts=16,
        top_k=2,
        backbone_structure="resnet18",
        backbone_pretrained=False,
        num_features=32,
        hidden_size=64,
        output_size=num_classes,
        per_token_noise=True,
        min_noise_scale=1e-3,
        router_temperature=1.2,
        capacity_factor=1.5,
        overflow_strategy="drop",
        bias_lr=0.1,
        ema_decay=0.9,
        bias_clip=2.0,
        update_bias_in_eval=False,
    ).to(device)
    return model


def build_base_moe(num_classes: int,
                   device: torch.device) -> nn.Module:
    """
    BASE MoE (balanced assignment).
    """
    model = BASE_Moe(
        num_experts=16,
        backbone_structure="resnet18",
        backbone_pretrained=False,
        num_features=32,
        hidden_size=64,
        output_size=num_classes,
        assign_mode="hungarian",     # falls back to greedy if SciPy is unavailable
        use_noisy_scores=False,
        min_noise_scale=1e-2,
    ).to(device)
    return model


def build_model(model_name: str,
                num_classes: int,
                device: torch.device,
                best_cfg: Dict[str, Any]) -> nn.Module:
    """
    Unified model builder by name.
    """
    model_name = model_name.lower()
    if model_name == "bayes":
        return build_bayesian_moe(num_classes, device, best_cfg)
    if model_name == "simple":
        return build_simple_moe(num_classes, device)
    if model_name == "ec":
        return build_expert_choice(num_classes, device)
    if model_name == "auxfree":
        return build_aux_free(num_classes, device)
    if model_name == "base":
        return build_base_moe(num_classes, device)
    raise ValueError(f"Unknown model type: {model_name}")


# ---------------------------------------------------------
# Training & evaluation loops (with Bayes warm-up)
# ---------------------------------------------------------

def _run_bayes_warmup(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
) -> None:
    """
    Internal helper: run warm-up epochs for Bayesian_NN_Moe.

    Priority:
      1) If model has train_one_epoch_warmup(...), use that (preferred).
      2) Otherwise, temporarily:
           - set beta_kl = 0 (disable KL)
           - set kl_anneal_steps = 0 (no annealing)
           - set router_mode = 'expected' (deterministic expectation routing)
         run standard train_one_epoch(...), then restore original values.
    """
    if warmup_epochs <= 0:
        return

    print(f"\n[Bayes] Warm-up: {warmup_epochs} epoch(s) before main training\n")

    # Case 1: explicit warm-up method provided by the model
    if hasattr(model, "train_one_epoch_warmup") and callable(getattr(model, "train_one_epoch_warmup")):
        for ep in range(1, warmup_epochs + 1):
            t0 = time.time()
            avg_loss, train_acc = model.train_one_epoch_warmup(
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                criterion=None,
                max_batches=None,
            )
            t1 = time.time()
            print(
                f"  [Warmup explicit] Epoch {ep:03d}: "
                f"train_loss={avg_loss:.4f} train_acc={train_acc:.4f} "
                f"time={t1 - t0:.1f}s"
            )
        return

    # Case 2: attribute-based warm-up (no special method)
    # Backup original attributes
    old_router_mode = getattr(model, "router_mode", None)
    old_beta_kl = getattr(model, "beta_kl", None)
    old_kl_anneal = getattr(model, "kl_anneal_steps", None)

    # Apply warm-up settings
    if hasattr(model, "beta_kl"):
        model.beta_kl = 0.0
    if hasattr(model, "kl_anneal_steps"):
        model.kl_anneal_steps = 0
    if hasattr(model, "router_mode"):
        model.router_mode = "expected"

    for ep in range(1, warmup_epochs + 1):
        t0 = time.time()
        avg_loss, train_acc = model.train_one_epoch(
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion=None,
            max_batches=None,
        )
        t1 = time.time()
        print(
            f"  [Warmup implicit] Epoch {ep:03d}: "
            f"train_loss={avg_loss:.4f} train_acc={train_acc:.4f} "
            f"time={t1 - t0:.1f}s"
        )

    # Restore original attributes
    if old_router_mode is not None and hasattr(model, "router_mode"):
        model.router_mode = old_router_mode
    if old_beta_kl is not None and hasattr(model, "beta_kl"):
        model.beta_kl = old_beta_kl
    if old_kl_anneal is not None and hasattr(model, "kl_anneal_steps"):
        model.kl_anneal_steps = old_kl_anneal


def train_model(model: nn.Module,
                train_loader: DataLoader,
                device: torch.device,
                epochs: int,
                lr: float,
                weight_decay: float,
                model_name: str,
                bayes_warmup_epochs: int = 0) -> None:
    """
    Train a MoE model using its own train_one_epoch(...) method.

    All models share the signature:
        train_one_epoch(loader, optimizer, device, criterion=None, max_batches=None)

    For Bayesian_NN_Moe (model_name == 'bayes'):
      - If bayes_warmup_epochs > 0, run a warm-up phase before main training:
          * Prefer model.train_one_epoch_warmup(...) if available.
          * Otherwise, temporarily disable KL and force 'expected' router_mode.

    Note: here `epochs` is treated as the TOTAL number of epochs,
    and the main phase uses max(epochs - bayes_warmup_epochs, 1).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Optional warm-up for Bayesian model
    if model_name.lower() == "bayes" and bayes_warmup_epochs > 0:
        _run_bayes_warmup(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            warmup_epochs=bayes_warmup_epochs,
        )

    # Main training loop
    main_epochs = max(epochs - bayes_warmup_epochs, 1)
    for ep in range(1, main_epochs + 1):
        t0 = time.time()
        avg_loss, train_acc = model.train_one_epoch(
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion=None,        # each model uses CE internally
            max_batches=None,
        )
        t1 = time.time()
        print(
            f"  [Main] Epoch {ep:03d}: train_loss={avg_loss:.4f} "
            f"train_acc={train_acc:.4f}  time={t1 - t0:.1f}s"
        )


@torch.no_grad()
def evaluate_moe(model: nn.Module,
                 test_loader: DataLoader,
                 device: torch.device,
                 num_experts: int,
                 num_classes: int) -> Dict[str, Any]:
    """
    Evaluate a MoE model on ID test set, computing:
      - accuracy, NLL, ECE
      - routing CV, NMI(expert; label), overflow rate, routing entropy
      - H_E_norm, H_Y_norm, EAS (entropy alignment)
      - DARS = 0.5 * EAS + 0.5 * NMI (data-aware routing score)

    We use model(x, return_aux=True) to gather routing statistics.
    This works for all MoE variants as long as they expose per_expert_counts,
    overflow_dropped, routing_entropy and topk_idx in aux.
    """
    model.eval()

    logits_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    counts = torch.zeros(num_experts, dtype=torch.float64)
    dropped = torch.zeros(num_experts, dtype=torch.float64)
    entropies: List[float] = []

    # for NMI
    expert_assign_list: List[np.ndarray] = []
    label_list: List[np.ndarray] = []

    for xb, yb in test_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits, aux = model(xb, return_aux=True)

        logits_all.append(logits.detach().cpu())
        labels_all.append(yb.detach().cpu())

        # per-expert counts & overflow
        pec = aux.get("per_expert_counts", None)
        ofd = aux.get("overflow_dropped", None)
        if pec is not None:
            counts += pec.detach().cpu().to(torch.float64)
        if ofd is not None:
            dropped += ofd.detach().cpu().to(torch.float64)

        # routing entropy
        if "routing_entropy" in aux:
            entropies.append(float(aux["routing_entropy"].detach().cpu().item()))

        # top-1 expert assignment
        tki = aux.get("topk_idx", None)
        if tki is not None:
            ha = tki[:, 0].detach().cpu().numpy()
            expert_assign_list.append(ha)
            label_list.append(yb.detach().cpu().numpy())

    if not logits_all:
        return {
            "acc": float("nan"),
            "nll": float("nan"),
            "ece": float("nan"),
            "cv": float("nan"),
            "nmi": float("nan"),
            "overflow": float("nan"),
            "routing_entropy": float("nan"),
            "H_E_norm": float("nan"),
            "H_Y_norm": float("nan"),
            "EAS": float("nan"),
            "DARS": float("nan"),
        }

    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)

    # classification metrics
    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    nll = F.cross_entropy(logits, labels).item()
    ece = ece_score(logits, labels)

    # routing metrics
    counts_np = counts.numpy()
    total_kept = counts_np.sum()
    total_dropped = dropped.numpy().sum()
    total_attempt = total_kept + total_dropped

    if total_kept > 0:
        cv = float(counts_np.std() / (counts_np.mean() + 1e-8))
        overflow_rate = float(total_dropped / (total_attempt + 1e-8))
    else:
        cv = float("nan")
        overflow_rate = float("nan")

    routing_entropy = float(np.mean(entropies)) if entropies else float("nan")

    # specialization NMI
    if expert_assign_list and label_list:
        ha = np.concatenate(expert_assign_list, axis=0)
        lb = np.concatenate(label_list, axis=0)
        nmi = specialization_nmi(ha, lb, num_experts)
    else:
        ha = np.array([], dtype=int)
        lb = np.array([], dtype=int)
        nmi = float("nan")

    # entropy alignment metrics between expert usage and label distribution
    H_E_norm, H_Y_norm, EAS = entropy_alignment_metrics(
        counts_np=counts_np,
        labels_np=labels.numpy(),
        num_experts=num_experts,
        num_classes=num_classes,
    )

    # Data-aware Routing Score: simple blend of entropy alignment and specialization
    # lambda = 0.5 -> equal weight on EAS and NMI
    if not (math.isnan(EAS) or math.isnan(nmi)):
        DARS = 0.5 * EAS + 0.5 * nmi
    else:
        DARS = float("nan")

    return dict(
        acc=float(acc),
        nll=float(nll),
        ece=float(ece),
        cv=cv,
        nmi=nmi,
        overflow=overflow_rate,
        routing_entropy=routing_entropy,
        H_E_norm=H_E_norm,
        H_Y_norm=H_Y_norm,
        EAS=EAS,
        DARS=DARS,
    )


@torch.no_grad()
def compute_confidence_stats(model: nn.Module,
                             loader: DataLoader,
                             device: torch.device) -> Dict[str, Any]:
    """
    Compute simple max-softmax confidence stats over a dataset.
    This is label-agnostic and suitable for OOD.
    """
    model.eval()
    softmax = nn.Softmax(dim=1)
    conf_list: List[torch.Tensor] = []

    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb, return_aux=False)
        probs = softmax(logits)
        max_conf, _ = probs.max(dim=1)
        conf_list.append(max_conf.detach().cpu())

    if not conf_list:
        return {
            "mean_conf": float("nan"),
            "std_conf": float("nan"),
            "n": 0,
        }

    all_conf = torch.cat(conf_list, dim=0).numpy()
    return {
        "mean_conf": float(all_conf.mean()),
        "std_conf": float(all_conf.std()),
        "n": int(all_conf.size),
    }


# ---------------------------------------------------------
# CLI parsing & main experiment loop
# ---------------------------------------------------------

def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Compare MoE models on CIFAR-10/100 with SVHN OOD")
    ap.add_argument("--dataset", type=str, default="cifar100",
                    choices=["cifar10", "cifar100"],
                    help="ID dataset to train on.")
    ap.add_argument("--models", type=str, default="bayes,simple,ec,auxfree,base",
                    help="Comma-separated list of models: bayes,simple,ec,auxfree,base")
    ap.add_argument("--ood", type=str, default="svhn",
                    help="OOD dataset name (svhn,stl10,cifar10,cifar100). Empty string to disable OOD.")
    ap.add_argument("--epochs", type=int, default=40,
                    help="Total training epochs (warm-up + main).")
    ap.add_argument("--bayes-warmup-epochs", type=int, default=0,
                    help="Number of warm-up epochs for Bayesian_NN_Moe before main training. "
                         "Ignored for non-Bayes models.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--best-config", type=str, default="best_config.json",
                    help="Path to best_config.json from Bayesian ablation script.")
    ap.add_argument("--device", type=str, default="auto",
                    help="'auto', 'cuda', or 'cpu'")
    ap.add_argument("--results-path", type=str, default="moe_compare_results.csv")
    return ap.parse_args()


def main():
    args = parse_args()

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # dataset
    train_loader, test_loader, num_classes = build_id_loaders(
        args.dataset, batch_size=args.batch_size,
        num_workers=args.workers, download=True
    )
    print(f"Built ID dataset: {args.dataset}, num_classes={num_classes}")

    # OOD loader (optional)
    ood_name = args.ood.strip().lower()
    ood_loader: Optional[DataLoader] = None
    if ood_name:
        try:
            ood_loader = build_ood_loader(
                ood_name, batch_size=args.batch_size,
                num_workers=args.workers, download=True
            )
            print(f"Built OOD dataset: {ood_name}")
        except Exception as e:
            print(f"[WARN] Failed to build OOD loader '{ood_name}': {e}")
            ood_loader = None

    # load best config for Bayesian model
    best_cfg = load_best_config(args.best_config)

    model_names = [s.strip().lower() for s in args.models.split(",") if s.strip()]
    print(f"Models to compare: {model_names}")

    results: List[Dict[str, Any]] = []

    for mname in model_names:
        print(f"\n================ Model: {mname} =================")
        # build model
        model = build_model(mname, num_classes, device, best_cfg)

        # train (with optional Bayes warm-up)
        train_model(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            model_name=mname,
            bayes_warmup_epochs=args.bayes_warmup_epochs,
        )

        # evaluate on ID
        ev = evaluate_moe(
            model=model,
            test_loader=test_loader,
            device=device,
            num_experts=16,        # fixed in all our configurations
            num_classes=num_classes,
        )

        # confidence stats (ID)
        id_stats = compute_confidence_stats(
            model=model,
            loader=test_loader,
            device=device,
        )

        # confidence stats (OOD)
        if ood_loader is not None:
            ood_stats = compute_confidence_stats(
                model=model,
                loader=ood_loader,
                device=device,
            )
        else:
            ood_stats = {
                "mean_conf": float("nan"),
                "std_conf": float("nan"),
                "n": 0,
            }

        row = dict(
            dataset=args.dataset,
            model=mname,
            epochs=args.epochs,
            bayes_warmup_epochs=(args.bayes_warmup_epochs if mname == "bayes" else 0),
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            # ID metrics
            test_acc=ev["acc"],
            test_nll=ev["nll"],
            ece=ev["ece"],
            cv=ev["cv"],
            nmi=ev["nmi"],
            overflow=ev["overflow"],
            routing_entropy=ev["routing_entropy"],
            H_E_norm=ev["H_E_norm"],
            H_Y_norm=ev["H_Y_norm"],
            EAS=ev["EAS"],
            DARS=ev["DARS"],
            id_mean_conf=id_stats["mean_conf"],
            id_std_conf=id_stats["std_conf"],
            id_n=id_stats["n"],
            # OOD metrics
            ood_dataset=ood_name if ood_loader is not None else "",
            ood_mean_conf=ood_stats["mean_conf"],
            ood_std_conf=ood_stats["std_conf"],
            ood_n=ood_stats["n"],
        )
        results.append(row)

        print(
            f"[{args.dataset}][{mname}] acc={row['test_acc']:.4f} "
            f"nll={row['test_nll']:.3f} ece={row['ece']:.4f} "
            f"cv={row['cv']:.3f} nmi={row['nmi']:.3f} overflow={row['overflow']:.4f} "
            f"H_E_norm={row['H_E_norm']:.3f} H_Y_norm={row['H_Y_norm']:.3f} "
            f"EAS={row['EAS']:.3f} DARS={row['DARS']:.3f} "
            f"id_mean_conf={row['id_mean_conf']:.4f} "
            f"ood_mean_conf={row['ood_mean_conf']:.4f}"
        )

    # save CSV
    df = pd.DataFrame(results)
    df.to_csv(args.results_path, index=False)
    print(f"\nSaved results to: {args.results_path}\n")

    # quick text summary
    cols = [
        "dataset", "model", "bayes_warmup_epochs",
        "test_acc", "test_nll", "ece",
        "cv", "nmi", "overflow",
        "H_E_norm", "H_Y_norm", "EAS", "DARS",
        "id_mean_conf", "ood_mean_conf",
    ]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
