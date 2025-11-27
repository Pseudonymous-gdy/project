import os
import sys

# Ensure project root is on sys.path when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# project-local
import cifar10
import cifar100
from Backbone_and_Expert import Backbone, Expert


class Aux_Free_Moe(nn.Module):
    """
    DeepSeek/PKU-style Auxiliary-Loss-Free MoE.

    Key idea:
      - Keep token-choice Top-k routing, BUT do NOT add any balancing auxiliary loss.
      - Instead, maintain a learnable-free (non-grad) expert-level bias vector 'expert_bias'.
      - After each training batch, update 'expert_bias' towards balanced usage using
        a simple online controller with an EMA of recent per-expert loads.

      Routing score per token:
          gate_scores = W_g * features + noise + expert_bias[None, :]
        where expert_bias is updated *outside* autograd.

    Features:
      - Per-token noise (optional) for Noisy Top-k.
      - Capacity factor (optional) with overflow counting (drop strategy).
      - Efficient dispatch: run only selected experts.
      - API compatible with your other MoE modules:
          forward(x, return_aux=True) -> (logits, aux)
          train_one_epoch / evaluate

    Acceleration (A100 / A800 / 4090 friendly):
      - Mixed precision training via torch.cuda.amp.autocast + GradScaler.
      - Non-blocking .to(device) transfers.
      - Optional channels-last memory format for 4D image tensors on CUDA.
    """

    def __init__(
        self,
        num_experts: int = 4,
        top_k: int = 1,
        *,
        # backbone / expert config
        backbone_structure: str = "resnet18",
        backbone_pretrained: bool = False,
        num_features: int = 32,
        hidden_size: int = 64,
        output_size: int = 10,
        # noisy gate
        per_token_noise: bool = True,
        min_noise_scale: float = 1e-3,
        router_temperature: float = 1.0,
        # capacity
        capacity_factor: float | None = None,    # e.g. 1.0/1.25, or None => unlimited
        overflow_strategy: str = "drop",         # currently only 'drop'
        # bias-balancer (no aux loss)
        bias_lr: float = 0.1,                    # update step for expert_bias
        ema_decay: float = 0.9,                  # EMA for measured per-expert load
        bias_clip: float | None = 2.0,           # clamp expert_bias to [-bias_clip, bias_clip]; None to disable
        update_bias_in_eval: bool = False,       # usually False; only update in train mode
        # AMP / mixed precision options
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,  # good default for modern NVIDIA GPUs
    ):
        super().__init__()
        assert num_experts >= 1
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))

        # Backbone: maps images -> feature vectors
        self.Back_bone = Backbone(
            structure=backbone_structure,
            pretrained=backbone_pretrained,
            num_features=num_features,
        )

        # Experts: each maps [B_e, F] -> [B_e, C]
        self.experts = nn.ModuleList(
            [
                Expert(
                    num_features=num_features,
                    hidden_size=hidden_size,
                    output_size=output_size,
                )
                for _ in range(self.num_experts)
            ]
        )

        # Gating linear: features -> expert scores
        self.gate_linear = nn.Linear(num_features, self.num_experts)

        # Noisy gate
        self.per_token_noise = bool(per_token_noise)
        self.min_noise_scale = float(min_noise_scale)
        if self.per_token_noise:
            self.noise_linear = nn.Linear(num_features, self.num_experts)
            # Mild noise init: softplus(-2) ~ 0.126
            nn.init.zeros_(self.noise_linear.weight)
            nn.init.constant_(self.noise_linear.bias, -2.0)

        self.router_temperature = float(router_temperature)

        # Capacity config
        self.capacity_factor = capacity_factor
        self.overflow_strategy = overflow_strategy

        # ---- aux-free balancer states (no grad) ----
        # expert_bias: added to gate scores; updated outside autograd
        self.register_buffer("expert_bias", torch.zeros(self.num_experts))
        # ema_load: EMA of recent per-expert *fractional* loads (sum to ~1)
        self.register_buffer(
            "ema_load",
            torch.full((self.num_experts,), 1.0 / self.num_experts),
        )

        self.bias_lr = float(bias_lr)
        self.ema_decay = float(ema_decay)
        self.bias_clip = bias_clip
        self.update_bias_in_eval = bool(update_bias_in_eval)

        # Cached output dim (derived from first expert's final linear)
        self._out_dim = self.experts[0].layer2.out_features

        # Mixed precision configuration
        self.use_amp = bool(use_amp) and torch.cuda.is_available()
        self.amp_dtype = amp_dtype
        self.scaler = GradScaler(enabled=self.use_amp)

    # -------------- helpers --------------

    def _compute_capacity(self, B: int) -> int | None:
        """
        Compute per-expert capacity for this micro-batch:

            cap = ceil(capacity_factor * (B * k / E))

        If capacity_factor is None, capacity is unlimited (return None).
        """
        if self.capacity_factor is None:
            return None
        fair_share = (B * self.top_k) / float(self.num_experts)
        cap = int(math.ceil(float(self.capacity_factor) * fair_share))
        return max(1, cap)

    @staticmethod
    def _safe_cv2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Coefficient of variation squared, with a small epsilon for stability.
        (Kept for potential diagnostics / analysis.)
        """
        m = x.mean()
        v = x.var(unbiased=False)
        return v / (m * m + eps)

    # -------------- forward (routing + dispatch) --------------

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        Forward pass:

          1) features = Backbone(x)
          2) gate_scores = W_g * features + noise + expert_bias
          3) Top-k selection per token; softmax over top-k for mixing
          4) Capacity handling; efficient dispatch; weighted combine
        """
        B = x.size(0)
        device = x.device

        # Optional channels-last layout for 4D image tensors on CUDA
        if x.dim() == 4 and x.is_cuda:
            x = x.to(memory_format=torch.channels_last)

        # 1) Features
        features = self.Back_bone(x)                             # [B, F]

        # 2) Scores (gating + noise + expert bias)
        # Align feature dtype with gate parameters for AMP safety
        f_gate = features
        p_dtype = self.gate_linear.weight.dtype
        if f_gate.dtype != p_dtype:
            f_gate = f_gate.to(p_dtype)
        raw_scores = self.gate_linear(f_gate)                    # [B, E]
        if self.per_token_noise:
            # Align dtype for noise linear as well
            f_noise = features
            p_dtype_n = self.noise_linear.weight.dtype
            if f_noise.dtype != p_dtype_n:
                f_noise = f_noise.to(p_dtype_n)
            sigma = F.softplus(self.noise_linear(f_noise)) + self.min_noise_scale
            scores = raw_scores + sigma * torch.randn_like(raw_scores)
        else:
            scores = raw_scores
        # Add expert-level bias (broadcast over batch)
        scores = scores + self.expert_bias.view(1, -1)           # [B, E]

        # 3) Top-k routing
        k = self.top_k
        topk_scores, topk_idx = torch.topk(scores, k, dim=1)     # [B, k], [B, k]
        topk_w = F.softmax(topk_scores / self.router_temperature, dim=1)  # [B, k]
        gate_probs = F.softmax(scores, dim=1)                    # [B, E] (for logging)

        # 4) Capacity + dispatch
        capacity = self._compute_capacity(B)
        per_expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        overflow_dropped = torch.zeros(self.num_experts, dtype=torch.long, device=device)

        out_dim = self._out_dim
        combined = features.new_zeros((B, out_dim))

        for e in range(self.num_experts):
            # Tokens routed to expert e in any of the k slots
            mask = (topk_idx == e).any(dim=1)                    # [B]
            if not mask.any():
                continue

            token_idx = torch.nonzero(mask, as_tuple=False).view(-1)  # [Be]
            Be = token_idx.numel()

            # Capacity handling
            if capacity is not None and Be > capacity:
                overflow = Be - capacity
                overflow_dropped[e] += overflow
                token_idx = token_idx[:capacity]
                Be = capacity

            f_e = features[token_idx]
            # Align expert input dtype with its parameters
            p_dtype_e = next(self.experts[e].parameters()).dtype
            if f_e.dtype != p_dtype_e:
                f_e = f_e.to(p_dtype_e)
            y_e = self.experts[e](f_e)                           # [Be, C]

            w = (topk_idx[token_idx] == e).float()               # [Be, k]
            w = (topk_w[token_idx] * w).sum(dim=1, keepdim=True) # [Be, 1]

            # Align accumulation dtype with combined for AMP safety
            if y_e.dtype != combined.dtype:
                y_e = y_e.to(combined.dtype)
            if w.dtype != combined.dtype:
                w = w.to(combined.dtype)
            combined.index_add_(0, token_idx, w * y_e)
            per_expert_counts[e] += Be

        if return_aux:
            routing_entropy = (-gate_probs.clamp_min(1e-12).log() * gate_probs).sum(dim=1).mean()
            aux = {
                "gate_scores": scores.detach(),
                "gate_probs": gate_probs.detach(),
                "topk_idx": topk_idx.detach(),
                "topk_scores": topk_scores.detach(),
                "topk_weights": topk_w.detach(),
                "per_expert_counts": per_expert_counts.detach(),
                "overflow_dropped": overflow_dropped.detach(),
                "capacity": (None if capacity is None else int(capacity)),
                "routing_entropy": routing_entropy.detach(),
                "expert_bias": self.expert_bias.detach().clone(),
                "ema_load": self.ema_load.detach().clone(),
            }
            return combined, aux

        return combined

    # -------------- bias updater (aux-free balancing) --------------

    @torch.no_grad()
    def _update_bias(self, per_expert_counts: torch.Tensor, k: int):
        """
        Online update of expert_bias to encourage balanced usage.

        Steps:
          - Convert per-expert counts to *fractions* over total assignments
            (≈ B * k after capacity).
          - Smooth with an exponential moving average (EMA).
          - Update bias in the direction (target - ema_load):
              * overused experts -> bias decreases
              * underused experts -> bias increases
        """
        total_assign = float(per_expert_counts.sum().item())
        if total_assign <= 0:
            return  # nothing processed

        frac = per_expert_counts.float() / total_assign           # [E], sum≈1
        # EMA over per-expert usage
        self.ema_load.mul_(self.ema_decay).add_((1.0 - self.ema_decay) * frac)

        target = 1.0 / self.num_experts
        delta = target - self.ema_load                            # positive -> increase bias
        self.expert_bias.add_(self.bias_lr * delta)               # in-place update

        if self.bias_clip is not None:
            self.expert_bias.clamp_(-self.bias_clip, self.bias_clip)

    # -------------- train / eval --------------

    def train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        criterion: nn.Module | None = None,
        max_batches: int | None = None,
    ):
        """
        One training epoch *without* any auxiliary balancing loss.

        Balancing is handled entirely by the online 'expert_bias' updates
        after each batch.

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

            # Move data to device; use channels_last for 4D images on CUDA.
            inputs = inputs.to(device, non_blocking=True)
            if inputs.dim() == 4 and inputs.is_cuda:
                inputs = inputs.to(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward + loss inside autocast when AMP is enabled.
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs, aux = self.forward(inputs, return_aux=True)
                loss = criterion(outputs, targets)

            # Backward and optimizer step
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Aux-free balancing: update bias using current batch stats
            self._update_bias(aux["per_expert_counts"], k=self.top_k)

            running_loss += float(loss.item()) * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
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
        Evaluation.

        By default, we do NOT update bias in eval (deterministic behavior).
        If you want continuous adaptation at test-time (rare), set
        `update_bias_in_eval=True` when constructing the model.

        We do not use AMP here by default; evaluation is already relatively cheap
        and @torch.no_grad() saves memory and compute.
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

            outputs, aux = self.forward(inputs, return_aux=True)

            if self.update_bias_in_eval:
                self._update_bias(aux["per_expert_counts"], k=self.top_k)

            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

        return correct / total if total > 0 else 0.0


# ------------------ Minimal quick test ------------------
if __name__ == "__main__":
    """
    Quick sanity test:
      - CIFAR-100 with aux-free balancing (no aux loss).
      - Switch to CIFAR-10 by replacing expert heads and rebuilding optimizer.

    Acceleration setup:
      - Enable cudnn.benchmark for faster convolutions on fixed input shapes.
      - If available, set float32 matmul precision to 'high' for better Tensor Core usage.
      - Enable mixed precision (bfloat16 by default) when CUDA is present.
    """
    print("Running basic unit test for Aux_Free_Moe...")
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Aux_Free_Moe(
        num_experts=16,
        top_k=2,
        backbone_structure="resnet18",
        backbone_pretrained=False,
        num_features=32,
        hidden_size=64,
        output_size=100,
        per_token_noise=True,
        min_noise_scale=1e-3,
        router_temperature=1.2,
        capacity_factor=1.5,
        overflow_strategy="drop",
        bias_lr=0.1,
        ema_decay=0.9,
        bias_clip=2.0,
        update_bias_in_eval=False,
        use_amp=True,             # enable AMP for GPUs that support it
        amp_dtype=torch.bfloat16, # robust choice for A100/A800/4090
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
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
                f"[CIFAR-100][AuxFree] epoch={epoch+1} "
                f"avg_loss={avg_loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}"
            )

    # Switch to CIFAR-10
    print("Running basic unit test for Aux_Free_Moe on CIFAR-10...")
    with torch.no_grad():
        for expert in model.experts:
            in_features = expert.layer2.in_features
            expert.layer2 = nn.Linear(in_features, 10).to(device)
        model._out_dim = 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
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
                f"[CIFAR-10][AuxFree] epoch={epoch+1} "
                f"avg_loss={avg_loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}"
            )

    print("Aux_Free_Moe basic unit test complete.")
