"""
Ablation runner for Bayesian_NN_Moe (Bayesian router + sparse Top-k).

What this script does
---------------------
1) Runs a compact sweep over router/ELBO and routing/balance hyperparameters.
2) Trains via the model's own train_one_epoch(...) (which implements ELBO, KL anneal, aux balance).
3) After training, probes routing on a few batches to estimate load-balance metrics (CV/Gini/Overflow/Entropy/NMI).
4) Evaluates accuracy, NLL, ECE on the test split.
5) Plots relationships (Acc–CV, ECE–beta_kl, Overflow–Capacity, Img/s–CV) and saves CSV/JSON summaries.
6) Picks a "best" config via a simple multi-objective score.

Assumptions about your model API
--------------------------------
- class Bayesian_NN_Moe(nn.Module):
    * forward(self, x: torch.Tensor, *, return_aux: bool=False)
      - If return_aux=True -> returns (combined_logits, aux_dict)
      - If return_aux=False -> returns combined_logits
    * train_one_epoch(self, loader, optimizer, device, criterion=None, max_batches=None)
      -> returns (avg_loss, train_accuracy)

- The aux dict from forward(..., return_aux=True) contains (as in your class):
    * 'per_expert_counts': LongTensor[E]
    * 'overflow_dropped' : LongTensor[E]
    * 'routing_entropy'  : scalar float tensor
    * 'topk_idx'         : LongTensor[B, k]
    * 'kl'               : scalar (router KL)

Notes
-----
- We DO NOT call forward(x, y=...) anywhere.
- Throughput is measured per epoch as dataset_size / epoch_wall_time (median across epochs).
"""

import os, sys, time, json, math, argparse, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Your repo modules
import cifar10
import cifar100
from Moe.Bayesian_NN_Moe import Bayesian_NN_Moe


# =========================================================
# Metrics
# =========================================================
@torch.no_grad()
def ece_score(logits: torch.Tensor, y: torch.Tensor, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE) with equal-width confidence bins.
    """
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(1)
    acc = (pred == y).float()

    bins = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.any():
            ece += torch.abs(acc[m].mean() - conf[m].mean()) * m.float().mean()
    return float(ece.item())


def gini_from_counts(n: np.ndarray) -> float:
    """
    Gini coefficient for nonnegative counts over experts:
        G = sum_{i,j} |n_i - n_j| / (2 * E * sum_i n_i)
    """
    n = np.asarray(n, dtype=np.float64)
    s = n.sum()
    if s <= 0:
        return 0.0
    diffsum = np.abs(n[:, None] - n[None, :]).sum()
    return float(diffsum / (2.0 * len(n) * s))


def specialization_nmi(hard_assign: np.ndarray, labels: np.ndarray, num_experts: int) -> float:
    """
    Lightweight NMI(X;Y) / max(H(X), H(Y)) without sklearn.
    X = expert id in [0..E-1], Y = class id.

    Args:
        hard_assign: (N,) predicted expert ids
        labels     : (N,) ground-truth class ids
        num_experts: E

    Returns:
        NMI in [0,1] or NaN if invalid.
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

    return float(I / max(Hx, Hy))


# =========================================================
# Data & Model
# =========================================================
def get_loaders(dataset: str = "cifar100",
                bs: int = 64,
                num_workers: int = 4,
                data_dir: str = "./data") -> Tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, test_loader) from your repo helpers.
    """
    if dataset.lower() == "cifar10":
        train, test, _ = cifar10.get_dataloaders(batch_size=bs, num_workers=num_workers,
                                                 data_dir=data_dir, download=False)
    else:
        train, test, _ = cifar100.get_dataloaders(batch_size=bs, num_workers=num_workers,
                                                  data_dir=data_dir, download=False)
    return train, test


def build_model(cfg: Dict[str, Any],
                output_size: int,
                device: str) -> torch.nn.Module:
    """
    Instantiate Bayesian_NN_Moe with constructor-aligned argument names.
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
        capacity_factor=cfg["capacity"],          # matches your constructor
        overflow_strategy="drop",

        # router training mode
        router_mode=cfg["router_mode"],           # {'expected','mc'}
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
# Routing Probe (post-training)
# =========================================================
@torch.no_grad()
def probe_routing_metrics(model: torch.nn.Module,
                          loader: DataLoader,
                          device: str,
                          num_experts: int,
                          max_batches: int = 40) -> Dict[str, float]:
    """
    Light probe over at most `max_batches` of the *train* loader to estimate balance.

    Collects:
      - per_expert_counts (sum across batches)
      - overflow_dropped  (sum across batches)
      - routing_entropy   (mean over batches)
      - specialization NMI using Top-1 expert assignment (from aux['topk_idx'][:,0])
    """
    model.eval()
    E = num_experts
    counts = torch.zeros(E, dtype=torch.float64)
    dropped = torch.zeros(E, dtype=torch.float64)
    entropies: List[float] = []
    hard_assign: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for bidx, (xb, yb) in enumerate(loader):
        if bidx >= max_batches:
            break
        xb = xb.to(device, non_blocking=True)
        logits, aux = model(xb, return_aux=True)  # deterministic expectation path with aux

        # per-expert counts & overflow
        pec = aux.get("per_expert_counts", None)
        ofd = aux.get("overflow_dropped", None)
        if pec is not None:
            counts += pec.detach().cpu().to(torch.float64)
        if ofd is not None:
            dropped += ofd.detach().cpu().to(torch.float64)

        # entropy
        if "routing_entropy" in aux:
            entropies.append(float(aux["routing_entropy"].detach().cpu().item()))

        # top-1 expert id for NMI
        tki = aux.get("topk_idx", None)
        if tki is not None:
            ha = tki[:, 0].detach().cpu().numpy()
            hard_assign.append(ha)
            labels.append(yb.numpy())

    # Aggregate
    counts_np = counts.numpy()
    total_kept = counts_np.sum()
    total_dropped = dropped.numpy().sum()
    total_attempt = total_kept + total_dropped

    cv = float(counts_np.std() / (counts_np.mean() + 1e-8)) if total_kept > 0 else np.nan
    gini = gini_from_counts(counts_np) if total_kept > 0 else np.nan
    eeu = float((counts_np > 0).mean()) if total_kept > 0 else np.nan
    overflow_rate = float(total_dropped / total_attempt) if total_attempt > 0 else np.nan
    routing_entropy = float(np.mean(entropies)) if len(entropies) > 0 else np.nan

    nmi = np.nan
    if len(hard_assign) > 0 and len(labels) > 0:
        ha = np.concatenate(hard_assign, axis=0)
        lb = np.concatenate(labels, axis=0)
        nmi = specialization_nmi(ha, lb, E)

    return dict(cv=cv, gini=gini, overflow=overflow_rate,
                routing_entropy=routing_entropy, eeu=eeu, nmi=nmi)


# =========================================================
# Evaluation
# =========================================================
@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: DataLoader,
             device: str) -> Dict[str, float]:
    """
    Deterministic evaluation using model(x, return_aux=False).
    """
    model.eval()
    logits_all, y_all = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb, return_aux=False)
        logits_all.append(logits.detach().cpu())
        y_all.append(yb.detach().cpu())

    logits = torch.cat(logits_all, 0)
    y = torch.cat(y_all, 0)

    acc = (logits.argmax(1) == y).float().mean().item()
    nll = F.cross_entropy(logits, y).item()
    ece = ece_score(logits, y)
    return dict(acc=acc, nll=nll, ece=ece)


# =========================================================
# Multi-objective score (to pick a "best" config)
# =========================================================
def score_row(r: Dict[str, Any]) -> float:
    """
    Higher is better (acc, img_s); lower is better (nll, ece, cv, gini, overflow).
    """
    s = 0.0
    s += 2.0 * r.get("acc", 0.0)                  # accuracy dominates
    s += 0.5 * r.get("img_s_norm", 0.0)           # normalized throughput
    s -= 0.5 * r.get("nll", 2.0)
    s -= 1.0 * r.get("ece", 0.1) * 10             # bring ECE to ~[0,2]
    s -= 0.5 * r.get("cv", 0.5)
    s -= 0.25 * r.get("gini", 0.3)
    s -= 0.5 * r.get("overflow", 0.0) * 100       # overflow in %
    return float(s)


# =========================================================
# Sweeps & defaults
# =========================================================
def default_base(output_size: int) -> Dict[str, Any]:
    """
    Reasonable starting point. Adjust epochs/seeds for debug vs paper-grade runs.
    """
    return dict(
        dataset="cifar100",
        batch_size=64, num_workers=4, epochs=40,
        seeds=[0, 1, 2],

        # model/backbone
        backbone="resnet18",
        hidden_size=64,
        num_experts=16,
        num_features=32,

        # routing & regularization
        top_k=2,
        tau=1.2,
        capacity=1.0,                  # capacity_factor
        router_mode="expected",        # 'expected' or 'mc'
        elbo_samples=2,                # used when router_mode=='mc'
        use_control_variate=True,
        prior_var=10.0,
        beta_kl=1e-6,
        kl_anneal=5000,

        # optimizer & optional aux balances
        weight_decay=0.02,
        w_importance=0.0,
        w_load=0.0,

        # outputs
        output_size=output_size
    )


def sweep_list() -> List[Dict[str, Any]]:
    """
    Phase-I ablation sweep (compact but informative).
    """
    return [
        {},  # baseline
        {"router_mode": "mc", "elbo_samples": 4},
        {"router_mode": "mc", "elbo_samples": 4, "use_control_variate": False},
        {"beta_kl": 0.0}, {"beta_kl": 1e-5},
        {"tau": 0.7}, {"tau": 2.0},
        {"top_k": 1}, {"top_k": 4},
        {"capacity": 0.5}, {"capacity": 2.0},
        {"weight_decay": 0.0},
        {"w_importance": 0.01, "w_load": 0.01},
    ]


# =========================================================
# Plot helpers (neutral matplotlib; no custom colors/styles)
# =========================================================
def plot_scatter(df, x: str, y: str, fname: str, xlabel: Optional[str] = None, ylabel: Optional[str] = None):
    plt.figure()
    plt.scatter(df[x], df[y], s=24)
    plt.xlabel(xlabel or x)
    plt.ylabel(ylabel or y)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_line(df, x: str, y: str, group: Optional[str] = None,
              fname: str = "plot.png", xlabel: Optional[str] = None, ylabel: Optional[str] = None):
    plt.figure()
    if group is None:
        df = df.sort_values(by=x)
        plt.plot(df[x], df[y], marker="o")
    else:
        for g, sub in df.groupby(group):
            sub = sub.sort_values(by=x)
            plt.plot(sub[x], sub[y], marker="o", label=str(g))
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
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # dataset
    if args.dataset.lower() == "cifar10":
        output_size = 10
        train_loader, test_loader = get_loaders("cifar10", bs=args.bs, num_workers=args.workers, data_dir=args.datadir)
    else:
        output_size = 100
        train_loader, test_loader = get_loaders("cifar100", bs=args.bs, num_workers=args.workers, data_dir=args.datadir)

    # base + CLI overrides
    base = default_base(output_size)
    base["dataset"] = args.dataset
    base["batch_size"] = args.bs
    base["num_workers"] = args.workers
    base["epochs"] = args.epochs
    base["seeds"] = [int(s) for s in args.seeds.split(",")]

    sweeps = sweep_list()
    results: List[Dict[str, Any]] = []

    for i, delta in enumerate(sweeps):
        cfg = {**base, **delta}

        for seed in cfg["seeds"]:
            # reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            # model & optimizer
            model = build_model(cfg, output_size, device)
            opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=cfg["weight_decay"])

            # -----------------------------
            # Train for cfg["epochs"] epochs
            # -----------------------------
            per_epoch_ips = []
            for ep in range(cfg["epochs"]):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()

                avg_loss, tr_acc = model.train_one_epoch(
                    loader=train_loader,
                    optimizer=opt,
                    device=device,
                    criterion=None,         # model uses CE internally if None
                    max_batches=None
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.time()

                # images/sec for this epoch (dataset-size based)
                num_imgs = len(train_loader.dataset)
                ips = num_imgs / max(t1 - t0, 1e-9)
                per_epoch_ips.append(ips)

            # -----------------------------
            # Evaluation
            # -----------------------------
            ev = evaluate(model, test_loader, device)

            # -----------------------------
            # Routing probe (post-training; few batches)
            # -----------------------------
            bal = probe_routing_metrics(
                model=model,
                loader=train_loader,
                device=device,
                num_experts=cfg["num_experts"],
                max_batches=40  # small probe; adjust if needed
            )

            # Throughput: robust median across epochs
            img_s = float(np.median(per_epoch_ips)) if len(per_epoch_ips) else np.nan

            row = dict(
                id=i, seed=seed, img_s=img_s,
                **ev, **bal, **cfg
            )
            results.append(row)
            print(
                f"[sweep={i} seed={seed}] "
                f"acc={row['acc']:.4f} nll={row['nll']:.3f} ece={row['ece']:.4f} "
                f"cv={row.get('cv', np.nan):.3f} gini={row.get('gini', np.nan):.3f} "
                f"overflow={row.get('overflow', np.nan):.4f} img/s={img_s:.1f}",
                flush=True
            )

    # -----------------------------
    # Persist & post-process
    # -----------------------------
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("ablation_results.csv", index=False)

    # Normalize img/s for scoring
    if df["img_s"].notna().any():
        mn, mx = df["img_s"].min(), df["img_s"].max()
        df["img_s_norm"] = (df["img_s"] - mn) / (max(mx - mn, 1e-9))
    else:
        df["img_s_norm"] = 0.0

    # Composite score & best config
    df["score"] = df.apply(score_row, axis=1)
    best_idx = int(df["score"].idxmax())
    best_row = df.iloc[best_idx].to_dict()

    with open("best_config.json", "w") as f:
        json.dump({
            k: best_row[k] for k in [
                "router_mode", "elbo_samples", "use_control_variate",
                "beta_kl", "kl_anneal", "tau", "top_k", "capacity",
                "weight_decay", "w_importance", "w_load",
                "score", "acc", "ece", "nll", "cv", "gini", "overflow", "img_s"
            ] if k in best_row
        }, f, indent=2)

    # -----------------------------
    # Plots
    # -----------------------------
    # (1) Accuracy vs CV
    if "cv" in df.columns:
        plot_scatter(df.dropna(subset=["cv"]), "cv", "acc",
                     "plot_acc_vs_cv.png",
                     xlabel="CV (lower = more balanced)", ylabel="Accuracy")

    # (2) ECE vs beta_kl (grouped by router_mode)
    if "beta_kl" in df.columns:
        plot_line(df, x="beta_kl", y="ece",
                  group="router_mode",
                  fname="plot_ece_vs_beta.png",
                  xlabel="beta_kl", ylabel="ECE")

    # (3) Overflow vs capacity
    if "capacity" in df.columns and "overflow" in df.columns:
        plot_line(df, x="capacity", y="overflow",
                  group=None,
                  fname="plot_overflow_vs_capacity.png",
                  xlabel="Capacity factor", ylabel="Overflow rate")

    # (4) Images/s vs CV (efficiency–balance relationship)
    if "img_s" in df.columns and "cv" in df.columns:
        plot_scatter(df.dropna(subset=["cv"]), "cv", "img_s",
                     "plot_imgs_vs_cv.png",
                     xlabel="CV (lower = more balanced)", ylabel="Images/sec")

    df.to_csv("ablation_results_scored.csv", index=False)
    print("\nSaved:")
    print("  - ablation_results.csv")
    print("  - ablation_results_scored.csv")
    print("  - best_config.json")
    print("  - plots: plot_acc_vs_cv.png, plot_ece_vs_beta.png, plot_overflow_vs_capacity.png, plot_imgs_vs_cv.png")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ablations for Bayesian_NN_Moe")
    ap.add_argument("--dataset", type=str, default="cifar100", choices=["cifar10", "cifar100"],
                    help="Dataset choice")
    ap.add_argument("--datadir", type=str, default="./data", help="Data directory")
    ap.add_argument("--bs", type=int, default=64, help="Batch size")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    ap.add_argument("--epochs", type=int, default=1, help="Training epochs per sweep variant")
    ap.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds, e.g. '0,1,2'")
    args = ap.parse_args()
    run(args)
