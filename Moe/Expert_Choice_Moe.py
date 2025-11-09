import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# project-local modules
import cifar10
import cifar100
from Backbone_and_Expert import Backbone, Expert


class Expert_Choice(nn.Module):
    """
    Expert Choice (EC) Mixture-of-Experts.

    Key idea:
      - "Experts pick tokens": for each expert e, select top-M tokens by that expert's
        column score. M = ceil(capacity_factor * B / E). This yields *hard* equalized load.
      - A token may be selected by multiple experts; we aggregate expert outputs by
        uniform averaging:  y = (1/m) * sum_{e in selected(token)} f_e(x), where m is
        the number of experts that selected this token.
      - No auxiliary balancing loss is needed; EC is balanced by construction.

    API is compatible with your Simple_Moe:
      - forward(x, return_aux=True) returns (logits, aux) where aux contains
        per_expert_counts, capacity, gate_probs, etc.
      - train_one_epoch / evaluate mirror your existing signatures.

    Notes:
      - For selection stability we use RAW gate scores (noisy selection optional).
      - Capacity factor drives the per-expert bucket size. If None, we default to 1.0.
      - All ops are AMP friendly; if you want AMP, just wrap training/evaluation with autocast.
    """

    def __init__(
        self,
        num_experts: int = 4,
        *,
        # backbone / expert sizes
        backbone_structure: str = "resnet18",
        backbone_pretrained: bool = False,
        num_features: int = 32,
        hidden_size: int = 64,
        output_size: int = 10,
        # EC routing knobs
        capacity_factor: float | None = 1.0,   # per-expert bucket multiplier; None -> 1.0
        use_noisy_scores: bool = False,        # if True, add per-token Gaussian noise to scores
        min_noise_scale: float = 1e-2,         # floor for noise scale
    ):
        super().__init__()
        assert num_experts >= 1
        self.num_experts = int(num_experts)

        # shared feature extractor
        self.Back_bone = Backbone(
            structure=backbone_structure,
            pretrained=backbone_pretrained,
            num_features=num_features,
        )

        # experts bank
        self.experts = nn.ModuleList(
            [Expert(num_features=num_features, hidden_size=hidden_size, output_size=output_size)
             for _ in range(self.num_experts)]
        )

        # gating linear (produces raw per-expert scores per token)
        self.gate_linear = nn.Linear(num_features, self.num_experts)

        # optional per-token noise for selection (usually kept False for EC)
        self.use_noisy_scores = bool(use_noisy_scores)
        self.min_noise_scale = float(min_noise_scale)
        if self.use_noisy_scores:
            self.noise_linear = nn.Linear(num_features, self.num_experts)
            # initialize to mild noise at start: softplus(-2) ~ 0.126
            nn.init.zeros_(self.noise_linear.weight)
            nn.init.constant_(self.noise_linear.bias, -2.0)

        # capacity config
        self.capacity_factor = capacity_factor  # None -> treated as 1.0

        # cached out_dim (read dynamically in forward for robustness)
        self._out_dim = self.experts[0].layer2.out_features

    # ----------------- utilities -----------------

    def _bucket_size(self, B: int) -> int:
        """Per-expert bucket size M = ceil(cap_factor * B / E)."""
        cap = 1.0 if self.capacity_factor is None else float(self.capacity_factor)
        E = self.num_experts
        fair = (B / float(E))
        M = int(math.ceil(cap * fair))
        return max(1, M)

    # ----------------- forward -------------------

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        Expert-Choice forward:
          1) features = Backbone(x)
          2) raw_scores = W_g * features; optionally add per-token noise for selection
          3) For each expert e, pick top-M tokens by scores[:, e]
          4) For tokens selected by multiple experts, aggregate with uniform weight 1/m
          5) Return combined logits, plus aux stats if requested
        """
        B = x.size(0)
        device = x.device

        # 1) features
        features = self.Back_bone(x)                 # [B, F]

        # 2) gating scores (raw) + optional noise for selection
        raw_scores = self.gate_linear(features)      # [B, E]
        if self.use_noisy_scores:
            sigma = F.softplus(self.noise_linear(features)) + self.min_noise_scale  # [B, E]
            sel_scores = raw_scores + sigma * torch.randn_like(raw_scores)          # [B, E]
        else:
            sel_scores = raw_scores

        # soft probs over experts (for logging; EC is independent of loading balance)
        gate_probs = F.softmax(raw_scores, dim=1)    # [B, E]

        # 3) EC selection: per-expert top-M over tokens
        E = self.num_experts
        M = self._bucket_size(B) # per-expert capacity
        selected = torch.zeros(B, E, dtype=torch.bool, device=device)
        per_expert_counts = torch.zeros(E, dtype=torch.long, device=device)
        overflow_dropped  = torch.zeros(E, dtype=torch.long, device=device)  # EC is 0 as placeholder

        for e in range(E):
            col = sel_scores[:, e]                   # [B]
            if M >= B:
                idx = torch.arange(B, device=device) # [B]
                Be = B
            else:
                idx = torch.topk(col, k=M, dim=0).indices  # [M]
                Be = M
            selected[idx, e] = True # set selected tokens
            per_expert_counts[e] = Be # number of tokens assigned to expert e

        # 4) aggregate: uniform 1/m
        out_dim = self.experts[0].layer2.out_features # read dynamically
        combined = features.new_zeros((B, out_dim)) # [B, out_dim], initialized to 0
        m = selected.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1], num experts that selected each token

        for e in range(E):
            token_idx = torch.nonzero(selected[:, e], as_tuple=False).view(-1)  # [Be], indices of tokens for expert e
            if token_idx.numel() == 0:
                continue
            y_e = self.experts[e](features[token_idx])                          # [Be, out_dim], expert outputs
            w_e = (1.0 / m[token_idx]).to(y_e.dtype)                            # [Be, 1], uniform weight 1/m
            combined[token_idx] += w_e * y_e # accumulate expert outputs

        # 5) aux
        if return_aux: # return aux stats for logging
            routing_entropy = (-gate_probs.clamp_min(1e-12).log() * gate_probs).sum(dim=1).mean()
            aux = {
                "gate_scores": raw_scores.detach(),         # we log RAW (selection used raw or noisy)
                "gate_probs": gate_probs.detach(),
                "topk_idx": torch.zeros(B, 1, dtype=torch.long, device=device),    # placeholder for API compat
                "topk_scores": torch.zeros(B, 1, device=device),
                "topk_weights": torch.ones(B, 1, device=device),
                "per_expert_counts": per_expert_counts.detach(),
                "overflow_dropped": overflow_dropped.detach(),
                "capacity": int(M),
                "routing_entropy": routing_entropy.detach(),
            }
            return combined, aux
        return combined

    # ----------------- train / eval -----------------

    def train_one_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                        device: torch.device, criterion: nn.Module | None = None,
                        max_batches: int | None = None):
        """
        EC training for one epoch. No balancing loss is used (EC is balanced by design).
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break # break if reached max batches for this epoch

            inputs = inputs.to(device, non_blocking=True) # [B, C, H, W]
            targets = targets.to(device, non_blocking=True) # [B]

            optimizer.zero_grad(set_to_none=True) # zero grads

            outputs, _aux = self.forward(inputs, return_aux=True) # forward pass
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # compute loss & accuracy stats
            running_loss += float(loss.item()) * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, device: torch.device, max_batches: int | None = None):
        """
        EC evaluation. For determinism you can keep use_noisy_scores=False (default).
        """
        self.eval()
        correct, total = 0, 0
        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = self.forward(inputs, return_aux=False)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
        return correct / total if total > 0 else 0.0


# ------------------ Minimal quick test ------------------
if __name__ == "__main__":
    """
    Quick sanity test:
      - CIFAR-100, capacity_factor=1.0 (per-expert bucket ~ B/E)
      - Switch to CIFAR-10 by replacing expert heads and rebuilding optimizer
    """
    print("Running basic unit test for Expert_Choice...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build EC model
    model = Expert_Choice(
        num_experts=16,
        backbone_structure="resnet18",
        backbone_pretrained=False,
        num_features=32,      # a bit larger feature for clearer routing
        output_size=100,      # CIFAR-100 first
        capacity_factor=1.0,  # typical EC bucket
        use_noisy_scores=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
    EPOCHS = 50

    # CIFAR-100 quick run
    train_loader, test_loader, _ = cifar100.get_dataloaders(
        batch_size=64, num_workers=0
    )
    for epoch in range(EPOCHS):
        avg_loss, acc = model.train_one_epoch(train_loader, optimizer, device, max_batches=20)
        if (epoch + 1) % 10 == 0:
            test_acc = model.evaluate(test_loader, device, max_batches=20)
            print(f"[CIFAR-100][EC] epoch={epoch+1} avg_loss={avg_loss:.4f} "
                  f"train_acc={acc:.4f} test_acc={test_acc:.4f}")

    # Switch to CIFAR-10 (replace expert heads and rebuild optimizer)
    print("Running basic unit test for Expert_Choice on CIFAR-10...")
    with torch.no_grad():
        # Update expert heads to 10-way
        for expert in model.experts:
            in_features = expert.layer2.in_features
            expert.layer2 = nn.Linear(in_features, 10).to(device)
        # (out_dim read dynamically in forward; no cache update needed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)

    train_loader, test_loader, _ = cifar10.get_dataloaders(
        batch_size=64, num_workers=0
    )
    for epoch in range(EPOCHS):
        avg_loss, acc = model.train_one_epoch(train_loader, optimizer, device, max_batches=20)
        if (epoch + 1) % 10 == 0:
            test_acc = model.evaluate(test_loader, device, max_batches=20)
            print(f"[CIFAR-10][EC] epoch={epoch+1} avg_loss={avg_loss:.4f} "
                  f"train_acc={acc:.4f} test_acc={test_acc:.4f}")

    print("Expert_Choice basic unit test complete.")
