import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
    ):
        super().__init__()
        assert num_experts >= 1
        self.num_experts = int(num_experts)
        self.top_k = max(1, min(int(top_k), self.num_experts))

        # backbone
        self.Back_bone = Backbone(
            structure=backbone_structure,
            pretrained=backbone_pretrained,
            num_features=num_features,
        )

        # experts
        self.experts = nn.ModuleList(
            [Expert(num_features=num_features, hidden_size=hidden_size, output_size=output_size)
             for _ in range(self.num_experts)]
        )

        # gating linear
        self.gate_linear = nn.Linear(num_features, self.num_experts)

        # noisy gate
        self.per_token_noise = bool(per_token_noise)
        self.min_noise_scale = float(min_noise_scale)
        if self.per_token_noise:
            self.noise_linear = nn.Linear(num_features, self.num_experts)
            # mild noise init: softplus(-2) ~ 0.126
            nn.init.zeros_(self.noise_linear.weight)
            nn.init.constant_(self.noise_linear.bias, -2.0)

        self.router_temperature = float(router_temperature)

        # capacity
        self.capacity_factor = capacity_factor
        self.overflow_strategy = overflow_strategy

        # ---- aux-free balancer states (no grad) ----
        # expert_bias: added to gate scores; will be updated outside autograd
        self.register_buffer("expert_bias", torch.zeros(self.num_experts))
        # ema_load: EMA of recent per-expert *fractional* loads (sum to ~1)
        self.register_buffer("ema_load", torch.full((self.num_experts,), 1.0 / self.num_experts))

        self.bias_lr = float(bias_lr)
        self.ema_decay = float(ema_decay)
        self.bias_clip = bias_clip
        self.update_bias_in_eval = bool(update_bias_in_eval)

    # -------------- helpers --------------

    def _compute_capacity(self, B: int) -> int | None:
        if self.capacity_factor is None:
            return None
        fair_share = (B * self.top_k) / float(self.num_experts)
        cap = int(math.ceil(float(self.capacity_factor) * fair_share))
        return max(1, cap)

    @staticmethod
    def _safe_cv2(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
        m = x.mean()
        v = x.var(unbiased=False)
        return v / (m * m + eps)

    # -------------- forward (routing + dispatch) --------------

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        1) features = Backbone(x)
        2) gate_scores = W_g * features + noise (+ expert_bias)
        3) Top-k selection per token; softmax over top-k for mixing
        4) Capacity handling; efficient dispatch; weighted combine
        """
        B = x.size(0)
        device = x.device

        # 1) features
        features = self.Back_bone(x)                             # [B, F]

        # 2) scores
        raw_scores = self.gate_linear(features)                  # [B, E]
        if self.per_token_noise:
            sigma = F.softplus(self.noise_linear(features)) + self.min_noise_scale
            scores = raw_scores + sigma * torch.randn_like(raw_scores)
        else:
            scores = raw_scores
        # add expert-level bias (broadcast over batch)
        scores = scores + self.expert_bias.view(1, -1)           # [B, E]

        # 3) Top-k route
        k = self.top_k
        topk_scores, topk_idx = torch.topk(scores, k, dim=1)     # [B, k], [B, k]
        topk_w = F.softmax(topk_scores / self.router_temperature, dim=1)  # [B, k]
        gate_probs = F.softmax(scores, dim=1)                     # [B, E] (for logging)

        # 4) capacity + dispatch
        capacity = self._compute_capacity(B)
        per_expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        overflow_dropped  = torch.zeros(self.num_experts, dtype=torch.long, device=device)

        out_dim = self.experts[0].layer2.out_features
        combined = features.new_zeros((B, out_dim))

        for e in range(self.num_experts):
            mask = (topk_idx == e).any(dim=1)                    # [B]
            if not mask.any():
                continue

            token_idx = torch.nonzero(mask, as_tuple=False).view(-1)  # [Be]
            Be = token_idx.numel()
            if capacity is not None and Be > capacity:
                overflow = Be - capacity
                overflow_dropped[e] += overflow
                token_idx = token_idx[:capacity]
                Be = capacity

            f_e = features[token_idx]
            y_e = self.experts[e](f_e)                           # [Be, C]

            w = (topk_idx[token_idx] == e).float()               # [Be, k]
            w = (topk_w[token_idx] * w).sum(dim=1, keepdim=True) # [Be, 1]

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
        - Convert counts to *fractions* over total assignments (≈B*k after capacity).
        - EMA the fractions to reduce noise.
        - Update bias in the direction (target - ema): if an expert is overused,
          reduce its bias; if underused, increase it.
        """
        total_assign = float(per_expert_counts.sum().item())
        if total_assign <= 0:
            return  # nothing processed

        frac = per_expert_counts.float() / total_assign           # [E], sum≈1
        # EMA
        self.ema_load.mul_(self.ema_decay).add_((1.0 - self.ema_decay) * frac)

        target = 1.0 / self.num_experts
        delta = (target - self.ema_load)                          # positive -> increase bias
        self.expert_bias.add_(self.bias_lr * delta)               # update in-place

        if self.bias_clip is not None:
            self.expert_bias.clamp_(-self.bias_clip, self.bias_clip)

    # -------------- train / eval --------------

    def train_one_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                        device: torch.device, criterion: nn.Module | None = None,
                        max_batches: int | None = None):
        """
        One training epoch *without* any aux balancing loss.
        Balancing is handled by the online 'expert_bias' updates after each batch.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs, aux = self.forward(inputs, return_aux=True)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # ---- aux-free balancing: update bias using current batch stats ----
            self._update_bias(aux["per_expert_counts"], k=self.top_k)

            running_loss += float(loss.item()) * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, device: torch.device, max_batches: int | None = None):
        """
        Evaluation. By default, we do NOT update bias in eval (deterministic behavior).
        If you want continuous adaptation at test-time (rare), set update_bias_in_eval=True.
        """
        self.eval()
        correct, total = 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = inputs.to(device, non_blocking=True)
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
    """
    print("Running basic unit test for Aux_Free_Moe...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Aux_Free_Moe(
        num_experts=6, top_k=1,
        backbone_structure="resnet18",
        backbone_pretrained=False,
        num_features=32, hidden_size=64, output_size=100,
        per_token_noise=True, min_noise_scale=1e-3,
        router_temperature=1.2,
        capacity_factor=1.25, overflow_strategy="drop",
        bias_lr=0.1, ema_decay=0.9, bias_clip=2.0,
        update_bias_in_eval=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
    EPOCHS = 30

    # CIFAR-100 quick run
    train_loader, test_loader, _ = cifar100.get_dataloaders(
        batch_size=64, num_workers=0)
    for epoch in range(EPOCHS):
        avg_loss, acc = model.train_one_epoch(train_loader, optimizer, device, max_batches=20)
        if (epoch + 1) % 10 == 0:
            test_acc = model.evaluate(test_loader, device, max_batches=20)
            print(f"[CIFAR-100][AuxFree] epoch={epoch+1} avg_loss={avg_loss:.4f} "
                  f"train_acc={acc:.4f} test_acc={test_acc:.4f}")

    # Switch to CIFAR-10
    print("Running basic unit test for Aux_Free_Moe on CIFAR-10...")
    with torch.no_grad():
        for expert in model.experts:
            in_features = expert.layer2.in_features
            expert.layer2 = nn.Linear(in_features, 10).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
    train_loader, test_loader, _ = cifar10.get_dataloaders(
        batch_size=64, num_workers=0
    )
    for epoch in range(EPOCHS):
        avg_loss, acc = model.train_one_epoch(train_loader, optimizer, device, max_batches=20)
        if (epoch + 1) % 10 == 0:
            test_acc = model.evaluate(test_loader, device, max_batches=20)
            print(f"[CIFAR-10][AuxFree] epoch={epoch+1} avg_loss={avg_loss:.4f} "
                  f"train_acc={acc:.4f} test_acc={test_acc:.4f}")

    print("Aux_Free_Moe basic unit test complete.")
