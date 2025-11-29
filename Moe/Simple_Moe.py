import os
import sys

# Ensure project root is on sys.path when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.amp import autocast, GradScaler

# Project-local modules (keep layout consistent with your repo)
import cifar10
import cifar100
from Backbone_and_Expert import Backbone, Expert


class Simple_Moe(nn.Module):
    """
    Shazeer-style Sparsely-Gated Mixture-of-Experts (MoE).

    Core features:
      - Shared Backbone to extract features of dimension `num_features`.
      - Gating linear layer producing per-expert scores.
      - Noisy Top-k routing (per-token noise by default).
      - Balanced load auxiliary losses: CV^2(importances) + CV^2(loads).
      - Optional capacity factor to limit per-expert tokens each step.
      - Efficient dispatch: only run selected experts on their assigned tokens.

    API:
      * forward(x, return_aux=False/True)
      * train_one_epoch(...)
      * evaluate(...)

    Acceleration (A100 / A800 / 4090 friendly):
      - Mixed precision training via torch.cuda.amp.autocast + GradScaler.
      - Non-blocking .to(device) in training / evaluation loops.
      - Optional channels-last memory format for 4D image tensors on CUDA.
    """

    def __init__(
        self,
        num_experts: int = 16,
        top_k: int = 2,
        aux_loss_weight: float = 0.0,      # kept for backward compat; prefer w_importance / w_load
        # AMP / mixed precision options
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,  # good default for modern NVIDIA GPUs
        **kwargs,
    ):
        super(Simple_Moe, self).__init__()

        # --------- public knobs (consistent with your usage) ----------
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))  # clamp k into [1, E]

        # Backbone configuration (same kw names as your original)
        num_features = kwargs.get("num_features", 32)
        output_size = kwargs.get("output_size", 10)
        self.Back_bone = Backbone(
            structure=kwargs.get("backbone_structure", "resnet18"),
            pretrained=kwargs.get("backbone_pretrained", False),
            num_features=num_features,
        )

        # Experts: each maps [B, num_features] -> [B, output_size]
        self.experts = nn.ModuleList(
            [
                Expert(
                    num_features=num_features,
                    hidden_size=kwargs.get("hidden_size", 64),
                    output_size=output_size,
                )
                for _ in range(self.num_experts)
            ]
        )

        # Gating linear layer: maps features -> E scores per token
        self.gate_linear = nn.Linear(num_features, self.num_experts)

        # ------------- Noisy Top-k configuration ----------------------
        # Per-token noise (Noisy Top-k style):
        #   score = W_g * feat + sigma(feat) * N(0,1)
        # If you prefer per-expert noise (token-invariant), set per_token_noise=False.
        self.per_token_noise = kwargs.get("per_token_noise", True)
        self.min_noise_scale = float(kwargs.get("min_noise_scale", 1e-2))
        if self.per_token_noise:
            # Learn sigma(feat) per token/expert
            self.noise_linear = nn.Linear(num_features, self.num_experts)
        else:
            # Learn a token-invariant scale per expert; broadcast across batch
            self.noise_scale = nn.Parameter(torch.zeros(self.num_experts))

        # Optional router softmax temperature over Top-k scores (stabilization knob)
        self.router_temperature = float(kwargs.get("router_temperature", 1.5))

        # ------------- Balancing loss weights ------------------------
        # Prefer explicit weights; if aux_loss_weight provided, split evenly.
        self.w_importance = float(kwargs.get("w_importance", 0.001))
        self.w_load = float(kwargs.get("w_load", 0.0005))
        if aux_loss_weight > 0.0:
            self.w_importance = self.w_importance or aux_loss_weight
            self.w_load = self.w_load or aux_loss_weight
        self.aux_loss_weight = float(aux_loss_weight)  # not used directly; kept for compatibility

        # ------------- Capacity factor (optional) ---------------------
        # If set, limit per-expert number of tokens per step to:
        #   cap = ceil(capacity_factor * (B * k / E))
        # Overflowed assignments are dropped (counted for monitoring).
        self.capacity_factor = kwargs.get("capacity_factor", None)  # e.g., 1.0/1.25/1.5 or None
        self.overflow_strategy = kwargs.get("overflow_strategy", "drop")  # currently 'drop' only

        # Cached output dim (derived from first expert's final linear)
        self._out_dim = self.experts[0].layer2.out_features

        # ------------- Mixed precision configuration ------------------
        # Enable AMP only when CUDA is available.
        self.use_amp = bool(use_amp) and torch.cuda.is_available()
        self.amp_dtype = amp_dtype
        # GradScaler safely scales the loss for fp16/bf16 backprop
        self.scaler = GradScaler(enabled=self.use_amp)

    # ======== Utilities for aux loss & capacity ========

    @staticmethod
    def _cv2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Coefficient of variation squared:
            CV^2(x) = Var(x) / (Mean(x)^2 + eps).

        This encourages balance without forcing exact uniformity.
        """
        mean = x.mean()
        var = x.var(unbiased=False)
        return var / (mean * mean + eps)

    def _compute_capacity(self, batch_size: int) -> int | None:
        """
        Capacity per expert for this micro-batch:

            cap = ceil(capacity_factor * (B * k / E))

        If capacity_factor is None, capacity is unlimited (return None).
        """
        if self.capacity_factor is None:
            return None
        E = self.num_experts
        k = self.top_k
        fair_share = (batch_size * k) / float(E)
        cap = int(math.ceil(float(self.capacity_factor) * fair_share))
        return max(1, cap)

    def _aux_balance_loss(
        self,
        gate_probs: torch.Tensor,            # [B, E] soft probs over experts
        topk_idx: torch.Tensor,              # [B, k] hard selected expert indices per token
        per_expert_counts: torch.Tensor | None = None,  # [E] post-capacity counts
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Balanced load auxiliary losses:

          - importance_e = sum_b p(b,e)                 (soft usage)
          - load_e       = #tokens actually processed   (hard usage, after capacity if provided)

        Total aux loss:
            L_aux = w_imp * CV^2(importance) + w_load * CV^2(load)
        """
        B, E = gate_probs.shape

        # Soft importance per expert: sum of probabilities across the batch.
        importance = gate_probs.sum(dim=0)     # [E]
        loss_imp = self._cv2(importance, eps) * self.w_importance

        # Hard load per expert:
        if per_expert_counts is not None:
            load = per_expert_counts.to(gate_probs.dtype)  # [E]
        else:
            # Approximation: count how many tokens selected each expert at least once (Top-k)
            onehot = torch.zeros(B, E, device=gate_probs.device, dtype=gate_probs.dtype)
            for j in range(topk_idx.size(1)):
                onehot.scatter_add_(1, topk_idx[:, j:j + 1], torch.ones_like(onehot[:, :1]))
            load = onehot.sum(dim=0)  # [E]
        loss_load = self._cv2(load, eps) * self.w_load

        return loss_imp + loss_load

    # ======== Forward: routing + efficient dispatch ========

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        Forward pass with Noisy Top-k routing and capacity-aware dispatch.

        Steps:
          1) Extract features using the shared backbone: [B, F].
          2) Compute per-expert gate scores and add Gaussian noise.
          3) Select Top-k experts per token; take softmax over the k scores
             (optionally with temperature) to obtain mixing weights.
          4) Capacity handling: limit how many tokens each expert can process.
          5) Efficient dispatch: only compute selected experts on their tokens.
          6) Weighted-sum combine the Top-k expert outputs for each token.

        If return_aux=True, also return a dict of routing statistics
        for logging and auxiliary balancing losses.
        """
        B = x.size(0)
        device = x.device

        # Shared features: [B, num_features]
        features = self.Back_bone(x)

        # Gating scores + Gaussian noise
        raw_scores = self.gate_linear(features)  # [B, E]
        if self.per_token_noise:
            sigma = F.softplus(self.noise_linear(features)) + self.min_noise_scale  # [B, E]
        else:
            sigma = self.noise_scale.view(1, -1).expand_as(raw_scores) + self.min_noise_scale
        gate_scores = raw_scores + sigma * torch.randn_like(raw_scores)             # [B, E]

        # Top-k selection per token
        k = self.top_k
        topk_scores, topk_indices = torch.topk(gate_scores, k, dim=1)              # [B, k], [B, k]
        # Mixing weights over the k selected experts (softmax with temperature)
        topk_weights = F.softmax(topk_scores / self.router_temperature, dim=1)     # [B, k]

        # Soft probs over all experts (for importance calculation and monitoring)
        gate_probs = F.softmax(gate_scores, dim=1)                                  # [B, E]

        # Capacity (optional)
        capacity = self._compute_capacity(B)  # int or None
        per_expert_counts = torch.zeros(self.num_experts, device=device, dtype=torch.long)
        overflow_dropped = torch.zeros(self.num_experts, device=device, dtype=torch.long)

        # Efficient dispatch to experts
        out_dim = self._out_dim  # determined from Expert.layer2 at __init__
        combined = features.new_zeros((B, out_dim))  # final logits per token

        # For each expert e, collect tokens assigned to it (by any of the k slots),
        # optionally apply capacity, run expert forward, and write back weighted outputs.
        for e in range(self.num_experts):
            # Mask tokens where expert e appears in any of the k positions
            mask = (topk_indices == e).any(dim=1)  # [B] bool
            if not mask.any():
                continue

            token_idx = torch.nonzero(mask, as_tuple=False).view(-1)  # [Be]
            Be = token_idx.numel()

            # Capacity handling: keep first `capacity` tokens, drop the rest
            if capacity is not None and Be > capacity:
                overflow = Be - capacity
                overflow_dropped[e] += overflow
                token_idx = token_idx[:capacity]
                Be = capacity

            # Features for tokens routed to expert e
            f_e = features[token_idx]                  # [Be, F]
            y_e = self.experts[e](f_e)                 # [Be, out_dim]

            # Compute per-token weight for THIS expert:
            # For each token b in token_idx, sum weights of positions where expert==e.
            # (In Top-k, expert appears at most once per token, so it's effectively one slot.)
            w = (topk_indices[token_idx] == e).float()             # [Be, k] indicator
            w = (topk_weights[token_idx] * w).sum(dim=1, keepdim=True)  # [Be, 1]

            # Weighted accumulation into final output
            combined[token_idx] += w * y_e

            # Track how many tokens expert e actually processed (post-capacity)
            per_expert_counts[e] += Be

        if return_aux:
            # Routing entropy (average per-token entropy over experts)
            routing_entropy = (-gate_probs.clamp_min(1e-12).log() * gate_probs).sum(dim=1).mean()

            aux = {
                "gate_scores": gate_scores.detach(),
                "gate_probs": gate_probs.detach(),
                "topk_idx": topk_indices.detach(),
                "topk_scores": topk_scores.detach(),
                "topk_weights": topk_weights.detach(),
                "per_expert_counts": per_expert_counts.detach(),
                "overflow_dropped": overflow_dropped.detach(),
                "capacity": (None if capacity is None else int(capacity)),
                "routing_entropy": routing_entropy.detach(),
            }
            return combined, aux

        return combined

    # ======== Training / Eval (keep your signatures) ========

    def train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        criterion: nn.Module | None = None,
        max_batches: int | None = None,
    ):
        """
        One training epoch that:
          - Calls forward(..., return_aux=True) ONCE to get both logits and routing stats.
          - Uses the SAME routing sample for the aux balancing loss (no re-sampling mismatch).
          - Tracks standard accuracy.

        Acceleration:
          - Mixed precision via autocast(enabled=self.use_amp, dtype=self.amp_dtype).
          - GradScaler for safe backprop when AMP is enabled.
          - Non-blocking transfers and optional channels-last for 4D inputs.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Move to device (GPU/CPU). For images on CUDA, channels_last can
            # improve performance via better memory access patterns.
            inputs = inputs.to(device, non_blocking=True)
            if inputs.dim() == 4 and inputs.is_cuda:
                inputs = inputs.to(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Single forward pass with aux routing stats, under autocast if AMP is enabled.
            with autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                outputs, aux = self.forward(inputs, return_aux=True)

                # Task loss
                ce = criterion(outputs, targets)

                # Balanced auxiliary loss (uses SAME routing sample)
                bal = self._aux_balance_loss(
                    gate_probs=aux["gate_probs"],
                    topk_idx=aux["topk_idx"],
                    per_expert_counts=aux["per_expert_counts"],
                )

                total_loss = ce + bal

            # Backward and optimizer step with GradScaler if AMP is enabled
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            # Accuracy & loss stats
            running_loss += float(total_loss.item()) * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        device: torch.device,
        max_batches: int | None = None,
    ):
        """
        Evaluation pass.

        By default, it keeps the same gating behavior as training
        (i.e., still uses noisy scores if self.per_token_noise=True).

        For strictly deterministic evaluation:
          - Temporarily set self.per_token_noise = False, or
          - Zero out the noise parameters (noise_linear / noise_scale).

        We do not use AMP here by default; evaluation is relatively cheap and
        @torch.no_grad() already saves memory and compute.
        """
        self.eval()
        correct, total = 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = inputs.to(device, non_blocking=True)
            if inputs.dim() == 4 and inputs.is_cuda:
                inputs = inputs.to(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)

            outputs = self.forward(inputs, return_aux=False)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        return correct / total if total > 0 else 0.0


# ------------------ Minimal quick test (optional) ------------------
if __name__ == "__main__":
    """
    Quick diagnostics:
      - Short run on CIFAR-100 with 100-way heads.
      - Then switch expert heads to 10-way for CIFAR-10 and rebuild optimizer.

    Acceleration setup:
      - Enable cudnn.benchmark for faster convolutions on fixed input shapes.
      - If available, set float32 matmul precision to 'high' to better leverage Tensor Cores.
      - Enable mixed precision (bfloat16 by default) when CUDA is present.
    """
    print("Running basic unit test for Simple_Moe...")
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Build model: 16 experts, Top-2 routing, per-token noise, CV-loss enabled
    model = Simple_Moe(
        num_experts=16,
        top_k=2,
        aux_loss_weight=0.0,
        backbone_structure="resnet18",
        backbone_pretrained=False,
        num_features=32,
        output_size=100,          # CIFAR-100
        per_token_noise=True,
        min_noise_scale=1e-2,
        w_importance=0.01,
        w_load=0.01,
        capacity_factor=1.5,
        overflow_strategy="drop",
        router_temperature=1.0,
        use_amp=True,             # AMP enabled for GPUs that support it
        amp_dtype=torch.bfloat16, # bfloat16 is robust on A100/A800/4090
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.02)
    EPOCHS = 50
    MAX_BATCHES_QTEST = 20  # keep small for quick smoke test

    # CIFAR-100 quick run
    train_loader, test_loader, _ = cifar100.get_dataloaders(
        batch_size=64,
        num_workers=0,
        data_dir="./data",
        download=False,
    )
    for epoch in range(EPOCHS):
        avg_loss, acc = model.train_one_epoch(
            train_loader,
            optimizer,
            device,
            max_batches=MAX_BATCHES_QTEST,
        )
        if (epoch + 1) % 10 == 0:
            test_acc = model.evaluate(
                test_loader,
                device,
                max_batches=MAX_BATCHES_QTEST,
            )
            print(
                f"[CIFAR-100] epoch={epoch+1} "
                f"avg_loss={avg_loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}"
            )

    # Switch experts to 10-way heads for CIFAR-10 and rebuild optimizer
    print("Running basic unit test for Simple_Moe on CIFAR-10...")
    with torch.no_grad():
        model._out_dim = 10
        for expert in model.experts:
            in_features = expert.layer2.in_features
            expert.layer2 = nn.Linear(in_features, 10).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.02)

    train_loader, test_loader, _ = cifar10.get_dataloaders(
        batch_size=64,
        num_workers=0,
        data_dir="./data",
        download=False,
    )
    for epoch in range(EPOCHS):
        avg_loss, acc = model.train_one_epoch(
            train_loader,
            optimizer,
            device,
            max_batches=MAX_BATCHES_QTEST,
        )
        if (epoch + 1) % 10 == 0:
            test_acc = model.evaluate(
                test_loader,
                device,
                max_batches=MAX_BATCHES_QTEST,
            )
            print(
                f"[CIFAR-10] epoch={epoch+1} "
                f"avg_loss={avg_loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}"
            )

    print("Simple_Moe basic unit test complete.")
