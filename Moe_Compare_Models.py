"""
Moe_Compare_Models_LinearLog.py

Compare multiple MoE variants on CIFAR-10 / CIFAR-100 with SVHN as OOD.

Models:
  - Bayesian_NN_Moe  (expected router, mc router; 4 presets total)
  - Simple_Moe       (Shazeer-style aux-loss MoE)
  - Expert_Choice    (EC MoE)
  - Aux_Free_Moe     (auxiliary-loss-free balancing)
  - BASE_Moe         (balanced assignment MoE)

Datasets (ID + OOD):
  - ID:  cifar10, cifar100
  - OOD: svhn  (via cifar100.get_ood_dataloader)

Metrics:
  - test_acc, test_nll, ECE
  - routing CV (per_expert_counts), NMI(expert; label), overflow rate
  - routing_entropy
  - ID / OOD max-softmax confidence stats (mean, std)

Bayes-aware extra metrics:
  - H_E_norm  : normalized entropy of expert usage
  - H_Y_norm  : normalized entropy of label distribution
  - EAS       : entropy alignment score 1 - |H_E_norm - H_Y_norm|
  - DARS      : Data-aware Routing Score = 0.5 * EAS + 0.5 * NMI

All training / evaluation progress is written linearly to:
  - stdout
  - moe_compare_train.txt
"""

import os
import sys
import math
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# =============================================================================
# Linear logging: everything goes to moe_compare_train.txt in chronological order
# =============================================================================

LOG_PATH = "moe_compare_train.txt"
_LOG_FH = None  # lazy open, so that file is created only when first used


def log(msg: str) -> None:
    """
    Simple linear logger:
      - prepends timestamp
      - prints to stdout
      - appends to moe_compare_train.txt
    """
    global _LOG_FH
    if _LOG_FH is None:
        _LOG_FH = open(LOG_PATH, "a", encoding="utf-8")

    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"{ts} | {msg}"
    print(line)
    _LOG_FH.write(line + "\n")
    _LOG_FH.flush()


# initial marker
log("==== Moe_Compare_Models (linear txt logging) started ====")

# =============================================================================
# Repo & model imports
# =============================================================================

# add repo root so that cifar10 / cifar100 / Moe.* can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cifar10
import cifar100

from Moe.Bayesian_NN_Moe import Bayesian_NN_Moe
from Moe.Simple_Moe import Simple_Moe
from Moe.Expert_Choice_Moe import Expert_Choice
from Moe.Aux_Free_Moe import Aux_Free_Moe
from Moe.BASE_Moe import BASE_Moe

# =============================================================================
# Global run configuration
# =============================================================================

NUM_EXPERTS = 16

# single global seed & single GPU
GLOBAL_SEED = 0

# default optimizer settings for non-Bayes models
DEFAULT_LR = 5e-4
DEFAULT_WEIGHT_DECAY = 2e-2

# main training epochs (warm-up epochs are extra)
EPOCHS = 40
BATCH_SIZE = 64
NUM_WORKERS = 2

# result file
RESULTS_CSV = "moe_compare_results.csv"

# which ID datasets & models to run
RUN_DATASETS = ["cifar10", "cifar100"]
RUN_SPECS = [
    dict(model="bayes", router_mode="expected"),
    dict(model="bayes", router_mode="mc"),
    dict(model="simple"),
    dict(model="ec"),
    dict(model="auxfree"),
    dict(model="base"),
]

# =============================================================================
# Four Bayes presets (per dataset, per router_mode)
#   - 2 套给 CIFAR-10 (expected / mc)
#   - 2 套给 CIFAR-100 (expected / mc)
# =============================================================================

BAYES_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    # ================= CIFAR-10 =================
    "cifar10": {
        # expected router: stable & fast
        "expected": dict(
            router_mode="expected",
            beta_kl=1e-6,
            tau=1.2,
            top_k=2,
            capacity=1.0,
            elbo_samples=2,
            use_control_variate=True,
            kl_anneal=5000,
            w_importance=0.0,
            w_load=0.0,
            warmup_epochs=5,
            lr=5e-4,
            weight_decay=1e-3,
        ),
        # mc ELBO: more Bayesian, needs stronger warm-up
        "mc": dict(
            router_mode="mc",
            beta_kl=1e-6,
            tau=1.2,
            top_k=2,
            capacity=1.0,
            elbo_samples=4,
            use_control_variate=True,
            kl_anneal=5000,
            w_importance=0.0,
            w_load=0.0,
            warmup_epochs=10,
            lr=5e-4,
            weight_decay=1e-3,
        ),
    },
    # ================= CIFAR-100 =================
    "cifar100": {
        # expected router preset from ECE / ACC / throughput analysis
        "expected": dict(
            router_mode="expected",
            beta_kl=1e-6,
            tau=1.2,
            top_k=2,
            capacity=1.0,
            elbo_samples=2,
            use_control_variate=True,
            kl_anneal=5000,
            w_importance=0.0,
            w_load=0.0,
            warmup_epochs=10,
            lr=5e-4,
            weight_decay=1e-3,
        ),
        # mc preset (needs warm-up)
        "mc": dict(
            router_mode="mc",
            beta_kl=1e-6,
            tau=1.2,
            top_k=2,
            capacity=1.0,
            elbo_samples=4,
            use_control_variate=True,
            kl_anneal=5000,
            w_importance=0.0,
            w_load=0.0,
            warmup_epochs=20,
            lr=5e-4,
            weight_decay=1e-3,
        ),
    },
}

# =============================================================================
# Utils: seeding, ECE, NMI, entropy alignment
# =============================================================================

def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    log(f"[Seed] Set global seed to {seed}")


@torch.no_grad()
def ece_score(logits: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
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
    x = np.asarray(hard_assign, dtype=int)
    y = np.asarray(labels, dtype=int)
    if x.size == 0:
        return float("nan")

    E = num_experts
    classes = np.unique(y)
    eps = 1e-12

    Px = np.array([(x == e).mean() for e in range(E)], dtype=np.float64) + eps
    Py = np.array([(y == c).mean() for c in classes], dtype=np.float64) + eps

    Hx = float(-(Px * np.log(Px)).sum())
    Hy = float(-(Py * np.log(Py)).sum())

    I = 0.0
    for i, e in enumerate(range(E)):
        for j, c in enumerate(classes):
            Pxy = float(((x == e) & (y == c)).mean()) + eps
            I += Pxy * (math.log(Pxy) - math.log(Px[i]) - math.log(Py[j]))
    return float(I / max(Hx, Hy))


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64) + 1e-12
    return float(-(p * np.log(p)).sum())


def entropy_alignment_metrics(counts_np: np.ndarray,
                              labels_np: np.ndarray,
                              num_experts: int,
                              num_classes: int) -> Tuple[float, float, float]:
    if counts_np.sum() <= 0 or labels_np.size == 0:
        return float("nan"), float("nan"), float("nan")

    p_e = counts_np.astype(np.float64) / counts_np.sum()
    H_E = _entropy(p_e)
    H_E_norm = H_E / max(math.log(num_experts + 1e-12), 1e-12)

    label_counts = np.bincount(labels_np.astype(int),
                               minlength=num_classes).astype(np.float64)
    if label_counts.sum() <= 0:
        return H_E_norm, float("nan"), float("nan")
    p_y = label_counts / label_counts.sum()
    H_Y = _entropy(p_y)
    H_Y_norm = H_Y / max(math.log(num_classes + 1e-12), 1e-12)

    eas = 1.0 - abs(H_E_norm - H_Y_norm)
    eas = float(max(0.0, min(1.0, eas)))
    return H_E_norm, H_Y_norm, eas


# =============================================================================
# Dataset builders (ID + OOD)
# =============================================================================

def build_id_loaders(dataset_name: str,
                     batch_size: int,
                     num_workers: int,
                     download: bool = True) -> Tuple[DataLoader, DataLoader, int]:
    dataset_name = dataset_name.lower()
    log(f"[Data] Building ID dataloaders for {dataset_name}")
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


def build_ood_loader_svhn(batch_size: int,
                          num_workers: int,
                          download: bool = True) -> DataLoader:
    log("[Data] Building OOD dataloader: SVHN")
    return cifar100.get_ood_dataloader(
        "svhn", batch_size=batch_size, num_workers=num_workers,
        data_dir="./data", download=download
    )


# =============================================================================
# Model builders
# =============================================================================

def build_bayesian_moe(num_classes: int,
                       device: torch.device,
                       cfg: Dict[str, Any]) -> nn.Module:
    base = dict(
        num_experts=NUM_EXPERTS,
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
    model_cfg = {**base, **cfg}

    log(
        "[Build] Bayesian_NN_Moe | "
        f"router_mode={model_cfg['router_mode']}  beta_kl={model_cfg['beta_kl']}  "
        f"tau={model_cfg['tau']}  top_k={model_cfg['top_k']}  "
        f"capacity={model_cfg['capacity']}  elbo_samples={model_cfg['elbo_samples']}"
    )

    model = Bayesian_NN_Moe(
        num_experts=model_cfg["num_experts"],
        num_features=model_cfg["num_features"],
        output_size=num_classes,
        top_k=model_cfg["top_k"],

        router_prior_var=model_cfg.get("prior_var", 10.0),
        beta_kl=model_cfg.get("beta_kl", 1e-6),
        kl_anneal_steps=model_cfg.get("kl_anneal", 5000),
        router_temperature=model_cfg.get("tau", 1.2),

        capacity_factor=model_cfg.get("capacity", 1.0),
        overflow_strategy="drop",

        router_mode=model_cfg.get("router_mode", "expected"),
        elbo_samples=model_cfg.get("elbo_samples", 2),
        use_control_variate=model_cfg.get("use_control_variate", True),

        backbone_structure="resnet18",
        backbone_pretrained=False,
        hidden_size=model_cfg["hidden_size"],

        w_importance=model_cfg.get("w_importance", 0.0),
        w_load=model_cfg.get("w_load", 0.0),
    ).to(device)
    return model


def build_simple_moe(num_classes: int, device: torch.device) -> nn.Module:
    log("[Build] Simple_Moe")
    model = Simple_Moe(
        num_experts=NUM_EXPERTS,
        top_k=2,
        aux_loss_weight=0.0,
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


def build_expert_choice(num_classes: int, device: torch.device) -> nn.Module:
    log("[Build] Expert_Choice_Moe")
    model = Expert_Choice(
        num_experts=NUM_EXPERTS,
        backbone_structure="resnet18",
        backbone_pretrained=False,
        num_features=32,
        hidden_size=64,
        output_size=num_classes,
        capacity_factor=1.0,
        use_noisy_scores=False,
    ).to(device)
    return model


def build_aux_free(num_classes: int, device: torch.device) -> nn.Module:
    log("[Build] Aux_Free_Moe")
    model = Aux_Free_Moe(
        num_experts=NUM_EXPERTS,
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


def build_base_moe(num_classes: int, device: torch.device) -> nn.Module:
    log("[Build] BASE_Moe")
    model = BASE_Moe(
        num_experts=NUM_EXPERTS,
        backbone_structure="resnet18",
        backbone_pretrained=False,
        num_features=32,
        hidden_size=64,
        output_size=num_classes,
        assign_mode="hungarian",
        use_noisy_scores=False,
        min_noise_scale=1e-2,
    ).to(device)
    return model


# =============================================================================
# Training & evaluation (with linear logging)
# =============================================================================

def _run_bayes_warmup(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
) -> None:
    if warmup_epochs <= 0:
        return

    log(f"[Bayes] Warm-up for {warmup_epochs} epoch(s)")

    if hasattr(model, "train_one_epoch_warmup") and callable(
        getattr(model, "train_one_epoch_warmup")
    ):
        for ep in range(1, warmup_epochs + 1):
            t0 = time.time()
            avg_loss, train_acc = model.train_one_epoch_warmup(
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                criterion=None,
                max_batches=None,
            )
            dt = time.time() - t0
            log(
                f"[Warmup explicit] epoch={ep:03d}  "
                f"loss={avg_loss:.4f}  acc={train_acc:.4f}  time={dt:.1f}s"
            )
        return

    old_router_mode = getattr(model, "router_mode", None)
    old_beta_kl = getattr(model, "beta_kl", None)
    old_kl_anneal = getattr(model, "kl_anneal_steps", None)

    if hasattr(model, "beta_kl"):
        model.beta_kl = 0.0
    if hasattr(model, "kl_anneal_steps"):
        model.kl_anneal_steps = 0
    if hasattr(model, "router_mode"):
        model.router_mode = "expected"

    log("[Bayes] Using implicit warm-up: beta_kl=0, router_mode='expected'")

    for ep in range(1, warmup_epochs + 1):
        t0 = time.time()
        avg_loss, train_acc = model.train_one_epoch(
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion=None,
            max_batches=None,
        )
        dt = time.time() - t0
        log(
            f"[Warmup implicit] epoch={ep:03d}  "
            f"loss={avg_loss:.4f}  acc={train_acc:.4f}  time={dt:.1f}s"
        )

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    log(
        f"[Train] model={model_name}  epochs(main)={epochs}  "
        f"warmup_epochs={bayes_warmup_epochs}  lr={lr}  wd={weight_decay}"
    )

    if model_name == "bayes" and bayes_warmup_epochs > 0:
        _run_bayes_warmup(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            warmup_epochs=bayes_warmup_epochs,
        )

    for ep in range(1, epochs + 1):
        t0 = time.time()
        avg_loss, train_acc = model.train_one_epoch(
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion=None,
            max_batches=None,
        )
        dt = time.time() - t0
        log(
            f"[Train-main] epoch={ep:03d}  "
            f"loss={avg_loss:.4f}  acc={train_acc:.4f}  time={dt:.1f}s"
        )


@torch.no_grad()
def evaluate_moe(model: nn.Module,
                 test_loader: DataLoader,
                 device: torch.device,
                 num_experts: int,
                 num_classes: int) -> Dict[str, Any]:
    log("[Eval] Evaluating on ID test set")
    model.eval()

    logits_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    counts = torch.zeros(num_experts, dtype=torch.float64)
    dropped = torch.zeros(num_experts, dtype=torch.float64)
    entropies: List[float] = []

    expert_assign_list: List[np.ndarray] = []
    label_list: List[np.ndarray] = []

    for xb, yb in test_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits, aux = model(xb, return_aux=True)

        logits_all.append(logits.detach().cpu())
        labels_all.append(yb.detach().cpu())

        pec = aux.get("per_expert_counts", None)
        ofd = aux.get("overflow_dropped", None)
        if pec is not None:
            counts += pec.detach().cpu().to(torch.float64)
        if ofd is not None:
            dropped += ofd.detach().cpu().to(torch.float64)

        if "routing_entropy" in aux:
            entropies.append(float(aux["routing_entropy"].detach().cpu().item()))

        tki = aux.get("topk_idx", None)
        if tki is not None:
            ha = tki[:, 0].detach().cpu().numpy()
            expert_assign_list.append(ha)
            label_list.append(yb.detach().cpu().numpy())

    if not logits_all:
        log("[Eval] Empty logits list; returning NaNs")
        return {k: float("nan") for k in
                ["acc", "nll", "ece", "cv", "nmi", "overflow",
                 "routing_entropy", "H_E_norm", "H_Y_norm", "EAS", "DARS"]}

    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)

    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    nll = F.cross_entropy(logits, labels).item()
    ece = ece_score(logits, labels)

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

    if expert_assign_list and label_list:
        ha = np.concatenate(expert_assign_list, axis=0)
        lb = np.concatenate(label_list, axis=0)
        nmi = specialization_nmi(ha, lb, num_experts)
    else:
        nmi = float("nan")

    H_E_norm, H_Y_norm, EAS = entropy_alignment_metrics(
        counts_np=counts_np,
        labels_np=labels.numpy(),
        num_experts=num_experts,
        num_classes=num_classes,
    )

    if not (math.isnan(EAS) or math.isnan(nmi)):
        DARS = 0.5 * EAS + 0.5 * nmi
    else:
        DARS = float("nan")

    log(
        "[Eval-ID] acc={:.4f}  nll={:.3f}  ece={:.4f}  "
        "cv={:.3f}  nmi={:.3f}  overflow={:.4f}  H_E_norm={:.3f}  "
        "H_Y_norm={:.3f}  EAS={:.3f}  DARS={:.3f}".format(
            acc, nll, ece, cv, nmi, overflow_rate,
            H_E_norm, H_Y_norm, EAS, DARS
        )
    )

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
                             device: torch.device,
                             tag: str) -> Dict[str, Any]:
    log(f"[Eval-{tag}] Computing max-softmax confidence stats")
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
        log(f"[Eval-{tag}] Empty loader; confidence stats NaN")
        return {"mean_conf": float("nan"), "std_conf": float("nan"), "n": 0}

    all_conf = torch.cat(conf_list, dim=0).numpy()
    mean_conf = float(all_conf.mean())
    std_conf = float(all_conf.std())
    n = int(all_conf.size)

    log(
        f"[Eval-{tag}] mean_conf={mean_conf:.4f}  std_conf={std_conf:.4f}  n={n}"
    )

    return {
        "mean_conf": mean_conf,
        "std_conf": std_conf,
        "n": n,
    }


# =============================================================================
# Main experiment loop (no CLI; run directly)
# =============================================================================

def main():
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    log(f"[Device] Using device: {device}")
    if device.type == "cuda":
        log(f"[Device] cuda.is_available()={torch.cuda.is_available()}")
        log(f"[Device] GPU count={torch.cuda.device_count()}")
        log(
            f"[Device] current device index={torch.cuda.current_device()} | "
            f"name={torch.cuda.get_device_name(0)}"
        )

    all_results: List[Dict[str, Any]] = []

    for ds in RUN_DATASETS:
        log("=" * 80)
        log(f"[Run] Dataset={ds} (ID) with SVHN as OOD")
        log("=" * 80)

        set_seed(GLOBAL_SEED)

        train_loader, test_loader, num_classes = build_id_loaders(
            ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, download=True
        )
        ood_loader = build_ood_loader_svhn(
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, download=True
        )

        for spec in RUN_SPECS:
            model_name = spec["model"]
            router_mode = spec.get("router_mode", "")

            set_seed(GLOBAL_SEED)  # ensure comparability

            if model_name == "bayes":
                cfg = BAYES_PRESETS[ds][router_mode]
                lr = cfg["lr"]
                weight_decay = cfg["weight_decay"]
                warmup_epochs = cfg["warmup_epochs"]

                log(
                    f"[Run] Model=Bayes router_mode={router_mode} on {ds} | "
                    f"beta_kl={cfg['beta_kl']}  tau={cfg['tau']}  "
                    f"top_k={cfg['top_k']}  capacity={cfg['capacity']}  "
                    f"elbo_samples={cfg['elbo_samples']}  "
                    f"warmup_epochs={warmup_epochs}  lr={lr}  wd={weight_decay}"
                )

                model = build_bayesian_moe(num_classes, device, cfg)
                train_model(
                    model=model,
                    train_loader=train_loader,
                    device=device,
                    epochs=EPOCHS,
                    lr=lr,
                    weight_decay=weight_decay,
                    model_name="bayes",
                    bayes_warmup_epochs=warmup_epochs,
                )

            elif model_name == "simple":
                log(f"[Run] Model=Simple_Moe on {ds}")
                model = build_simple_moe(num_classes, device)
                train_model(
                    model=model,
                    train_loader=train_loader,
                    device=device,
                    epochs=EPOCHS,
                    lr=DEFAULT_LR,
                    weight_decay=DEFAULT_WEIGHT_DECAY,
                    model_name="simple",
                    bayes_warmup_epochs=0,
                )

            elif model_name == "ec":
                log(f"[Run] Model=Expert_Choice on {ds}")
                model = build_expert_choice(num_classes, device)
                train_model(
                    model=model,
                    train_loader=train_loader,
                    device=device,
                    epochs=EPOCHS,
                    lr=DEFAULT_LR,
                    weight_decay=DEFAULT_WEIGHT_DECAY,
                    model_name="ec",
                    bayes_warmup_epochs=0,
                )

            elif model_name == "auxfree":
                log(f"[Run] Model=Aux_Free_Moe on {ds}")
                model = build_aux_free(num_classes, device)
                train_model(
                    model=model,
                    train_loader=train_loader,
                    device=device,
                    epochs=EPOCHS,
                    lr=DEFAULT_LR,
                    weight_decay=DEFAULT_WEIGHT_DECAY,
                    model_name="auxfree",
                    bayes_warmup_epochs=0,
                )

            elif model_name == "base":
                log(f"[Run] Model=BASE_Moe on {ds}")
                model = build_base_moe(num_classes, device)
                train_model(
                    model=model,
                    train_loader=train_loader,
                    device=device,
                    epochs=EPOCHS,
                    lr=DEFAULT_LR,
                    weight_decay=DEFAULT_WEIGHT_DECAY,
                    model_name="base",
                    bayes_warmup_epochs=0,
                )

            else:
                raise ValueError(f"Unknown model spec: {model_name}")

            # ===== Evaluation (ID + OOD) =====
            ev = evaluate_moe(
                model=model,
                test_loader=test_loader,
                device=device,
                num_experts=NUM_EXPERTS,
                num_classes=num_classes,
            )
            id_stats = compute_confidence_stats(model, test_loader, device, tag="ID")
            ood_stats = compute_confidence_stats(model, ood_loader, device, tag="OOD")

            if model_name == "bayes":
                row_lr = cfg["lr"]
                row_wd = cfg["weight_decay"]
                row_warm = cfg["warmup_epochs"]
                row_router_mode = router_mode
            else:
                row_lr = DEFAULT_LR
                row_wd = DEFAULT_WEIGHT_DECAY
                row_warm = 0
                row_router_mode = ""

            row = dict(
                dataset=ds,
                model=model_name,
                bayes_router_mode=row_router_mode,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=row_lr,
                weight_decay=row_wd,
                bayes_warmup_epochs=row_warm,
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
                ood_dataset="svhn",
                ood_mean_conf=ood_stats["mean_conf"],
                ood_std_conf=ood_stats["std_conf"],
                ood_n=ood_stats["n"],
            )
            all_results.append(row)

            log(
                "[Summary] [{ds}][{model}] acc={acc:.4f}  nll={nll:.3f}  "
                "ece={ece:.4f}  cv={cv:.3f}  nmi={nmi:.3f}  overflow={ov:.4f}  "
                "H_E_norm={He:.3f}  H_Y_norm={Hy:.3f}  EAS={eas:.3f}  "
                "DARS={dars:.3f}  id_mean_conf={idc:.4f}  "
                "ood_mean_conf={odc:.4f}".format(
                    ds=ds,
                    model=(model_name if model_name != "bayes"
                           else f"bayes-{router_mode}"),
                    acc=row["test_acc"],
                    nll=row["test_nll"],
                    ece=row["ece"],
                    cv=row["cv"],
                    nmi=row["nmi"],
                    ov=row["overflow"],
                    He=row["H_E_norm"],
                    Hy=row["H_Y_norm"],
                    eas=row["EAS"],
                    dars=row["DARS"],
                    idc=row["id_mean_conf"],
                    odc=row["ood_mean_conf"],
                )
            )

    # 保存 & 打印总表
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_CSV, index=False)
    log(f"[Result] Saved results to {RESULTS_CSV}")

    cols = [
        "dataset", "model", "bayes_router_mode", "bayes_warmup_epochs",
        "test_acc", "test_nll", "ece",
        "cv", "nmi", "overflow",
        "H_E_norm", "H_Y_norm", "EAS", "DARS",
        "id_mean_conf", "ood_mean_conf",
    ]
    log("\n" + df[cols].to_string(index=False))
    log("==== Moe_Compare_Models finished ====")

    # close log file
    global _LOG_FH
    if _LOG_FH is not None:
        _LOG_FH.close()
        _LOG_FH = None


if __name__ == "__main__":
    main()
