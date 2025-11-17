import os
import sys

# Ensure project root is on sys.path when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision  # kept for potential extensions / transforms
from torch.cuda.amp import autocast, GradScaler

# Optional exact solver (Hungarian algorithm via SciPy)
try:
    from scipy.optimize import linear_sum_assignment

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# project-local modules
import cifar10
import cifar100
from Backbone_and_Expert import Backbone, Expert


class BASE_Moe(nn.Module):
    """
    BASE (Balanced Assignment of Sparse Experts) Mixture-of-Experts.

    Core idea:
      - Treat routing as a (balanced) linear assignment problem:
          * Each token must be assigned to exactly one expert.
          * Per-expert loads are balanced: capacities differ by at most 1.
          * Maximize the total routing score.

    Implementation:
      - Build score matrix S in R^{B x E} from the backbone features.
      - Compute per-expert capacities that sum to B (balanced).
      - Solve an assignment that respects capacities and maximizes sum of scores.
        * 'hungarian': If SciPy is available, duplicate expert columns by capacity
          to get a square (B x B) matrix and solve exactly with Hungarian.
        * 'greedy'   : Fallback approximate solver: sort all (token, expert) pairs
          by score desc and fill tokens & capacities greedily.
      - Dispatch tokens only to their assigned expert and combine outputs (top-1).

    API is aligned with other MoE classes:
      - forward(x, return_aux=True) -> (logits, aux)
      - train_one_epoch(...) / evaluate(...)

    Acceleration (same style as Bayesian_NN_Moe / Expert_Choice):
      - Mixed precision training via torch.cuda.amp.autocast + GradScaler
      - Non-blocking .to(device) transfers
      - Optional channels-last memory format for 4D image tensors on CUDA
    """

    def __init__(
        self,
        num_experts: int = 4,
        *,
        # backbone / expert config
        backbone_structure: str = "resnet18",
        backbone_pretrained: bool = False,
        num_features: int = 32,
        hidden_size: int = 64,
        output_size: int = 10,
        # routing options
        assign_mode: str = "hungarian",   # 'hungarian' uses SciPy if available else falls back
        use_noisy_scores: bool = False,   # whether to add per-token Gaussian noise to routing scores
        min_noise_scale: float = 1e-2,    # minimum sigma if noise is used (stability)
        # AMP / mixed precision options
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,  # good default for A100/A800/4090
    ):
        super().__init__()
        assert num_experts >= 1
        self.num_experts = int(num_experts)

        # Shared Backbone: maps images -> feature vectors of length num_features
        # (e.g., a ResNet where we replace the final fc with a smaller linear layer).
        self.Back_bone = Backbone(
            structure=backbone_structure,
            pretrained=backbone_pretrained,
            num_features=num_features,
        )

        # Experts: a ModuleList of Expert modules. Each Expert maps [B_e, F] -> [B_e, C]
        # where C is the output dimension (e.g., number of classes).
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

        # Gating linear: maps features -> E scores for each token
        # Shape: [B, E] after calling gate_linear(features)
        self.gate_linear = nn.Linear(num_features, self.num_experts)

        # Optional per-token noise generator used to implement Noisy-TopK style routing.
        # If enabled, we compute sigma(features) and add sigma * N(0,1) to raw scores.
        self.use_noisy_scores = bool(use_noisy_scores)
        self.min_noise_scale = float(min_noise_scale)
        if self.use_noisy_scores:
            # noise_linear produces a per-token per-expert scale; softplus ensures positivity
            self.noise_linear = nn.Linear(num_features, self.num_experts)
            # Initialize to small sigma: set weights=0 and bias negative so softplus(bias) is small
            nn.init.zeros_(self.noise_linear.weight)
            nn.init.constant_(self.noise_linear.bias, -2.0)  # softplus(-2) ~ 0.126

        # Assignment solver selection: 'hungarian' if SciPy available gives exact optimum,
        # otherwise 'greedy' is a fast and simple fallback.
        self.assign_mode = assign_mode
        if self.assign_mode not in ("hungarian", "greedy"):
            raise ValueError("assign_mode must be 'hungarian' or 'greedy'")

        # Mixed precision configuration
        # Enabled only when CUDA is present.
        self.use_amp = bool(use_amp) and torch.cuda.is_available()
        self.amp_dtype = amp_dtype
        # GradScaler safely scales the loss for fp16/bf16 backprop
        self.scaler = GradScaler(enabled=self.use_amp)

    # ----------------- capacities -----------------

    def _balanced_capacities(self, B: int) -> torch.Tensor:
        """
        Compute integer capacities per expert such that:
          - capacities.sum() == B
          - capacities differ by at most 1 (most balanced split)

        Example:
          B=10, E=4 => q=2, r=2 => capacities = [3,3,2,2]

        Return:
          torch.LongTensor of shape [E] on CPU (move to device as needed).
        """
        E = self.num_experts
        q, r = divmod(B, E)
        caps = torch.full((E,), q, dtype=torch.long)
        if r > 0:
            caps[:r] += 1
        return caps  # [E], sum == B

    # ----------------- assignment solvers -----------------

    def _assign_hungarian(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Exact balanced assignment using the Hungarian algorithm (SciPy).

        Args:
          scores: [B, E] tensor of scores (higher -> better to assign token->expert)

        Procedure:
          - Compute per-expert capacities (caps) that sum to B.
          - Duplicate each expert column by its capacity to produce a [B, B] matrix.
          - Solve min-cost assignment on cost = -scores_expanded to maximize total score.
          - Map the chosen expanded column index back to the original expert id.

        Returns:
          assigned_e: [B] torch.LongTensor of expert indices for each token.

        Note:
          If SciPy isn't available we fallback to the greedy solver.
        """
        if not SCIPY_AVAILABLE:
            # Fallback to greedy if SciPy is unavailable
            return self._assign_greedy(scores)

        device = scores.device
        B, E = scores.shape
        caps = self._balanced_capacities(B).tolist()  # python ints required below

        # Build expanded column list and remember which expert each expanded column maps to
        exp_cols = []
        col2expert = []
        for e in range(E):
            # Duplicate column e, caps[e] times
            for _ in range(caps[e]):
                exp_cols.append(scores[:, e].unsqueeze(1))  # [B,1]
                col2expert.append(e)
        scores_exp = torch.cat(exp_cols, dim=1)  # [B, B]

        # Hungarian solves a minimum cost problem; we want maximum score.
        cost = (-scores_exp).detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)

        # Map expanded columns back to expert ids and return on the original device
        col2expert = torch.tensor(col2expert, dtype=torch.long)
        chosen_exp_cols = torch.tensor(col_ind, dtype=torch.long)  # [B]
        assigned_e = col2expert[chosen_exp_cols]                    # [B]
        return assigned_e.to(device)

    def _assign_greedy(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Greedy approximate balanced assignment.

        Procedure:
          - Flatten the [B, E] score matrix to a list of pairs (b,e).
          - Sort all pairs by score descending.
          - Iterate through sorted list, assign token b to expert e if token is free
            and expert e still has capacity.

        Guarantees a full assignment because total capacity == B.
        Complexity:
          O(B*E log(B*E)) due to sorting; OK for moderate sizes.
        """
        device = scores.device
        B, E = scores.shape
        caps = self._balanced_capacities(B).to(device)     # [E]
        token_assigned = torch.full((B,), -1, dtype=torch.long, device=device)  # -1 means free

        # Flatten and sort scores descending
        flat = scores.reshape(-1)
        order = torch.argsort(flat, descending=True)

        # Iterate until all tokens are assigned
        remaining = B
        for idx in order:
            if remaining == 0:
                break
            b = (idx // E).item()
            e = (idx % E).item()
            if token_assigned[b] != -1:
                continue
            if caps[e] <= 0:
                continue
            token_assigned[b] = e
            caps[e] -= 1
            remaining -= 1

        # If any tokens remain unassigned (very unlikely), assign them to experts with spare capacity
        if remaining > 0:
            for b in range(B):
                if token_assigned[b] == -1:
                    e = torch.argmax(caps).item()
                    token_assigned[b] = e
                    caps[e] -= 1
                    remaining -= 1
                    if remaining == 0:
                        break
        return token_assigned  # [B]

    # ----------------- forward -------------------

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        """
        BASE forward:

          1) features = Backbone(x)
          2) raw_scores = W_g * features (+ optional per-token noise)
          3) Solve balanced assignment: each token -> exactly one expert
          4) Dispatch tokens to their assigned expert; combine outputs (top-1)

        Notes:
          - For 4D image inputs on CUDA, we convert to channels_last memory format
            to allow cuDNN / Tensor Cores to be more efficient.
        """
        B = x.size(0)
        device = x.device

        # Optional channels-last layout for 4D image tensors on CUDA
        if x.dim() == 4 and x.is_cuda:
            x = x.to(memory_format=torch.channels_last)

        # 1) Features from shared backbone
        features = self.Back_bone(x)                       # [B, F]

        # 2) Routing scores
        raw_scores = self.gate_linear(features)            # [B, E]
        if self.use_noisy_scores:
            sigma = F.softplus(self.noise_linear(features)) + self.min_noise_scale
            scores = raw_scores + sigma * torch.randn_like(raw_scores)
        else:
            scores = raw_scores

        # For logging (BASE uses top-1 assignment, but we still track soft probs)
        gate_probs = F.softmax(raw_scores, dim=1)          # [B, E]

        # 3) Balanced assignment
        if self.assign_mode == "hungarian":
            assigned_e = self._assign_hungarian(scores)    # [B]
        else:
            assigned_e = self._assign_greedy(scores)       # [B]

        # 4) Efficient dispatch (top-1)
        out_dim = self.experts[0].layer2.out_features
        combined = features.new_zeros((B, out_dim))
        per_expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=device)

        # Vectorized grouping by expert: for each e, gather tokens and run the expert once
        for e in range(self.num_experts):
            token_idx = torch.nonzero(assigned_e == e, as_tuple=False).view(-1)  # [Be]
            if token_idx.numel() == 0:
                continue
            y_e = self.experts[e](features[token_idx])                           # [Be, C]
            # Accumulate with index_add_ (faster than 'combined[token_idx] += ...')
            combined.index_add_(0, token_idx, y_e)
            per_expert_counts[e] = token_idx.numel()

        if return_aux:
            routing_entropy = (-gate_probs.clamp_min(1e-12).log() * gate_probs).sum(dim=1).mean()
            # In BASE there is no overflow; keep field for API compatibility
            overflow_dropped = torch.zeros(self.num_experts, dtype=torch.long, device=device)
            caps = self._balanced_capacities(B)
            aux = {
                "gate_scores": raw_scores.detach(),
                "gate_probs": gate_probs.detach(),
                # BASE uses top-1 assignment; we reuse these fields for compatibility
                "topk_idx": assigned_e.view(B, 1).detach(),     # assigned expert as 'top-1'
                "topk_scores": torch.gather(scores, 1, assigned_e.view(-1, 1)).detach(),
                "topk_weights": torch.ones(B, 1, device=device),  # top-1 => weight 1
                "per_expert_counts": per_expert_counts.detach(),
                "overflow_dropped": overflow_dropped.detach(),
                "capacity": int(caps.sum().item()),  # equals B
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
        One training epoch for BASE.

        Objective:
          - Standard cross-entropy on top of BASE outputs.
          - No auxiliary balancing loss is needed (BASE is balanced by construction).

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

            # Non-blocking device transfer; channels-last for 4D images on CUDA.
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
        Evaluation; deterministic by default unless you enable noisy scores.

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
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
        return correct / total if total > 0 else 0.0


# ------------------ Minimal quick test ------------------
if __name__ == "__main__":
    """
    Quick sanity test:

      (A) CIFAR-100, balanced assignment (exact if SciPy is available, else greedy).
      (B) Switch to CIFAR-10 by replacing expert heads and rebuilding optimizer.

    Acceleration setup (same pattern as other MoE variants):
      - Enable cudnn.benchmark for faster convolutions on fixed input shapes.
      - If available, set float32 matmul precision to 'high' for better Tensor Core usage.
      - Enable mixed precision (bfloat16 by default) in the model when CUDA is present.
    """
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Running basic unit test for BASE_Moe...")

    # Build BASE model for CIFAR-100
    model = BASE_Moe(
        num_experts=16,
        backbone_structure="resnet18",
        backbone_pretrained=False,
        num_features=64,      # keep in line with EC quick test
        hidden_size=32,
        output_size=100,      # CIFAR-100 first
        assign_mode="hungarian",  # will fallback to 'greedy' if SciPy not found
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
                f"[CIFAR-100][BASE] epoch={epoch+1} "
                f"avg_loss={avg_loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}"
            )

    # Switch to CIFAR-10 (replace expert heads and rebuild optimizer)
    print("Running basic unit test for BASE_Moe on CIFAR-10...")
    with torch.no_grad():
        for expert in model.experts:
            in_features = expert.layer2.in_features
            expert.layer2 = nn.Linear(in_features, 10).to(device)
        # out_dim is read dynamically in forward; no cache update needed

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
                f"[CIFAR-10][BASE] epoch={epoch+1} "
                f"avg_loss={avg_loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}"
            )

    print("BASE_Moe basic unit test complete.")
