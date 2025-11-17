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

    Acceleration (same style as Bayesian_NN_Moe):
      - Mixed precision training via torch.cuda.amp.autocast + GradScaler
      - Non-blocking .to(device) transfers
      - Optional channels-last memory format for 4D image tensors
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
        # AMP / mixed precision options
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,  # good default for A100/A800/4090
    ):
        super().__init__()
        assert num_experts >= 1
        self.num_experts = int(num_experts)

        # Shared feature extractor (deterministic backbone)
        self.Back_bone = Backbone(
            structure=backbone_structure,
            pretrained=backbone_pretrained,
            num_features=num_features,
        )

        # Experts bank
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

        # Gating linear (produces raw per-expert scores per token)
        self.gate_linear = nn.Linear(num_features, self.num_experts)

        # Optional per-token noise for selection (usually kept False for EC)
        self.use_noisy_scores = bool(use_noisy_scores)
        self.min_noise_scale = float(min_noise_scale)
        if self.use_noisy_scores:
            self.noise_linear = nn.Linear(num_features, self.num_experts)
            # Initialize to mild noise at start: softplus(-2) ~ 0.126
            nn.init.zeros_(self.noise_linear.weight)
            nn.init.constant_(self.noise_linear.bias, -2.0)

        # Capacity config
        self.capacity_factor = capacity_factor  # None -> treated as 1.0

        # Cached out_dim (read dynamically in forward for robustness)
        self._out_dim = self.experts[0].layer2.out_features

        # Mixed precision configuration
        # Enabled only when CUDA is present.
        self.use_amp = bool(use_amp) and torch.cuda.is_available()
        self.amp_dtype = amp_dtype
        # GradScaler safely scales the loss for fp16/bf16 backprop
        self.scaler = GradScaler(enabled=self.use_amp)

    # ----------------- utilities -----------------

    def _bucket_size(self, B: int) -> int:
        """
        Compute per-expert bucket size:

            M = ceil(capacity_factor * B / E)

        where:
            B = batch size (#tokens),
            E = number of experts.
        """
        cap = 1.0 if self.capacity_factor is None else float(self.capacity_factor)
        E = self.num_experts
        fair = B / float(E)
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

        Note:
          - We optionally convert 4D image tensors to channels_last on CUDA
            to unlock cudnn/Tensor Core optimizations.
        """
        B = x.size(0)
        device = x.device

        # Channels-last layout for images on CUDA can improve throughput
        if x.dim() == 4 and x.is_cuda:
            x = x.to(memory_format=torch.channels_last)

        # 1) Features from shared backbone
        features = self.Back_bone(x)                 # [B, F]

        # 2) Gating scores (raw) + optional noise for selection
        raw_scores = self.gate_linear(features)      # [B, E]
        if self.use_noisy_scores:
            sigma = F.softplus(self.noise_linear(features)) + self.min_noise_scale  # [B, E]
            sel_scores = raw_scores + sigma * torch.randn_like(raw_scores)         # [B, E]
        else:
            sel_scores = raw_scores

        # Soft probabilities over experts (for logging; EC does not use them for routing)
        gate_probs = F.softmax(raw_scores, dim=1)    # [B, E]

        # 3) EC selection: per-expert top-M over tokens
        E = self.num_experts
        M = self._bucket_size(B)                     # per-expert capacity
        selected = torch.zeros(B, E, dtype=torch.bool, device=device)
        per_expert_counts = torch.zeros(E, dtype=torch.long, device=device)
        # EC always fills exactly M tokens per expert (no overflow), but we keep this for API symmetry
        overflow_dropped = torch.zeros(E, dtype=torch.long, device=device)

        for e in range(E):
            col = sel_scores[:, e]                   # [B]
            if M >= B:
                idx = torch.arange(B, device=device) # [B]
                Be = B
            else:
                idx = torch.topk(col, k=M, dim=0).indices  # [M]
                Be = M
            selected[idx, e] = True
            per_expert_counts[e] = Be

        # 4) Aggregate expert outputs with uniform 1/m weights
        out_dim = self.experts[0].layer2.out_features
        combined = features.new_zeros((B, out_dim))  # [B, out_dim]
        # m[b] = number of experts that selected token b
        m = selected.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]

        for e in range(E):
            token_idx = torch.nonzero(selected[:, e], as_tuple=False).view(-1)  # [Be]
            if token_idx.numel() == 0:
                continue
            # Expert outputs for selected tokens
            y_e = self.experts[e](features[token_idx])                          # [Be, out_dim]
            # Uniform weight 1/m for each token
            w_e = (1.0 / m[token_idx]).to(y_e.dtype)                            # [Be, 1]
            combined[token_idx] += w_e * y_e

        # 5) Aux dictionary (for logging / analysis)
        if return_aux:
            routing_entropy = (-gate_probs.clamp_min(1e-12).log() * gate_probs).sum(dim=1).mean()
            B = x.size(0)
            aux = {
                "gate_scores": raw_scores.detach(),         # raw gate scores (without noise)
                "gate_probs": gate_probs.detach(),
                # EC has no per-token Top-k in the usual sense; keep placeholders for API compatibility
                "topk_idx": torch.zeros(B, 1, dtype=torch.long, device=device),
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

    def train_one_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        criterion: nn.Module | None = None,
        max_batches: int | None = None,
    ):
        """
        Train EC for one epoch.

        Objective:
          - Standard cross-entropy on top of Expert-Choice outputs.
          - No balancing loss is used (EC is load-balanced by construction).

        Acceleration:
          - Mixed precision via autocast(enabled=self.use_amp, dtype=self.amp_dtype).
          - GradScaler for safe backprop when AMP is enabled.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Non-blocking transfer; channels-last for 4D image tensors.
            inputs = inputs.to(device, non_blocking=True)
            if inputs.dim() == 4 and inputs.is_cuda:
                inputs = inputs.to(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward + loss inside autocast when AMP is enabled.
            with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs, _aux = self.forward(inputs, return_aux=True)
                loss = criterion(outputs, targets)

            # Backward and optimizer step
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Accumulate statistics
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
        EC evaluation.

        For determinism you typically keep use_noisy_scores=False (default).
        We do not use AMP here by default; evaluation is already cheap relative to training.
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
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
        return correct / total if total > 0 else 0.0


# ------------------ Minimal quick test ------------------
if __name__ == "__main__":
    """
    Quick sanity test:

      (A) CIFAR-100 with Expert-Choice routing, capacity_factor=1.0
      (B) Switch to CIFAR-10 by replacing expert heads and rebuilding optimizer

    Acceleration setup (same style as Bayesian_NN_Moe):
      - Enable cudnn.benchmark for faster convolutions on fixed input shapes.
      - If available, set float32 matmul precision to 'high' for better Tensor Core usage.
      - Enable mixed precision (bfloat16) in the model when CUDA is present.
    """
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Running basic unit test for Expert_Choice...")

    # Build EC model for CIFAR-100
    model = Expert_Choice(
        num_experts=16,
        backbone_structure="resnet18",
        backbone_pretrained=False,
        num_features=32,      # a bit larger feature for clearer routing
        hidden_size=64,
        output_size=100,      # CIFAR-100 first
        capacity_factor=1.0,  # typical EC bucket
        use_noisy_scores=False,
        use_amp=True,              # enable AMP for GPUs that support it
        amp_dtype=torch.bfloat16,  # good default for A100/A800/RTX 4090
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
    EPOCHS = 50
    MAX_BATCHES_QTEST = 20  # keep small for quick smoke test

    # CIFAR-100 quick run
    train_loader, test_loader, _ = cifar100.get_dataloaders(
        batch_size=64, num_workers=0, data_dir="./data", download=False
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
                f"[CIFAR-100][EC] epoch={epoch+1} "
                f"avg_loss={avg_loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}"
            )

    # Switch to CIFAR-10 (replace expert heads and rebuild optimizer)
    print("Running basic unit test for Expert_Choice on CIFAR-10...")
    with torch.no_grad():
        # Update expert heads to 10-way
        for expert in model.experts:
            in_features = expert.layer2.in_features
            expert.layer2 = nn.Linear(in_features, 10).to(device)
        # _out_dim is read dynamically from experts in forward; no cache update needed

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)

    train_loader, test_loader, _ = cifar10.get_dataloaders(
        batch_size=64, num_workers=0, data_dir="./data", download=False
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
                f"[CIFAR-10][EC] epoch={epoch+1} "
                f"avg_loss={avg_loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}"
            )

    print("Expert_Choice basic unit test complete.")
