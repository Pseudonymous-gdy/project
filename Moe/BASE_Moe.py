import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

# optional exact solver
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
    Facebook: 'BASE' Mixture of Experts.
    Treat the routing as a (balanced) linear assignment problem:
      - Each token must be assigned to exactly one expert.
      - Per-expert loads are balanced: capacities differ by at most 1.
      - Maximize the total routing score.

    Implementation:
      - Build score matrix S in R^{B x E} from the backbone features.
      - Compute per-expert capacities that sum to B (balanced).
      - Solve an assignment that respects capacities and maximizes sum of scores.
        * 'hungarian': If SciPy is available, duplicate expert columns by capacity
           to get a square (B x B) matrix and solve exactly with Hungarian.
        * 'greedy':    Fallback approximate solver: sort all (token, expert) pairs
           by score desc and fill tokens & capacities greedily.
      - Dispatch tokens only to their assigned expert and combine outputs.

    API is aligned with your other MoE classes:
      - forward(x, return_aux=True) -> (logits, aux)
      - train_one_epoch / evaluate
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
        use_noisy_scores: bool = False,     # whether to add per-token Gaussian noise to routing scores
        min_noise_scale: float = 1e-2,      # minimum sigma if noise is used (stability)
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
            [Expert(num_features=num_features, hidden_size=hidden_size, output_size=output_size)
             for _ in range(self.num_experts)]
        )

        # Gating linear: maps features -> E scores for each token
        # Shape: [B, E] after calling gate_linear(features)
        self.gate_linear = nn.Linear(num_features, self.num_experts)

        # Optional per-token noise generator used to implement Noisy-TopK style routing
        # If enabled, we compute sigma(features) and add sigma * N(0,1) to raw scores.
        self.use_noisy_scores = bool(use_noisy_scores)
        self.min_noise_scale = float(min_noise_scale)
        if self.use_noisy_scores:
            # noise_linear produces a per-token per-expert scale; softplus ensures positivity
            self.noise_linear = nn.Linear(num_features, self.num_experts)
            # initialize to small sigma: set weights=0 and bias negative so softplus(bias) small
            nn.init.zeros_(self.noise_linear.weight)
            nn.init.constant_(self.noise_linear.bias, -2.0)  # softplus(-2) ~ 0.126

        # Assignment solver selection: 'hungarian' if SciPy available gives exact optimum,
        # otherwise 'greedy' is a fast and simple fallback.
        self.assign_mode = assign_mode
        if self.assign_mode not in ("hungarian", "greedy"):
            raise ValueError("assign_mode must be 'hungarian' or 'greedy'")

    # ----------------- capacities -----------------

    def _balanced_capacities(self, B: int) -> torch.Tensor:
        """
        Compute integer capacities per expert such that:
          - capacities.sum() == B
          - capacities differ by at most 1 (most balanced split)

        Example: B=10, E=4 => q=2, r=2 => capacities = [3,3,2,2]
        Return: torch.LongTensor of shape [E]
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

        Note: If SciPy isn't available we fallback to the greedy solver above.
        """
        if not SCIPY_AVAILABLE:
            # fallback to greedy if SciPy is unavailable
            return self._assign_greedy(scores)

        device = scores.device
        B, E = scores.shape
        caps = self._balanced_capacities(B).tolist()  # python ints required below

        # Build expanded column list and remember which expert each expanded column maps to
        exp_cols = []
        col2expert = []
        for e in range(E):
            # duplicate column e, caps[e] times
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
        Complexity: O(B*E log(B*E)) due to sorting; OK for modest sizes.
        """
        device = scores.device
        B, E = scores.shape
        caps = self._balanced_capacities(B).to(device)     # [E]
        token_assigned = torch.full((B,), -1, dtype=torch.long, device=device)  # -1 means free
        # Flatten and sort
        # idx_flat in [0, B*E), map to (b = //E, e = %E)
        flat = scores.reshape(-1)
        # argsort descending
        order = torch.argsort(flat, descending=True)
        # Iterate until all tokens are assigned
        remaining = B
        for idx in order:
            if remaining == 0:
                break
            b = (idx // E).item()
            e = (idx %  E).item()
            if token_assigned[b] != -1:
                continue
            if caps[e] <= 0:
                continue
            token_assigned[b] = e
            caps[e] -= 1
            remaining -= 1

        # If any tokens remain unassigned (shouldn't happen), assign them to experts with spare cap
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
          2) raw_scores = W_g * features (+ optional noise)
          3) Solve balanced assignment: each token -> exactly one expert
          4) Dispatch tokens to their assigned expert; combine outputs (top-1)
        """
        B = x.size(0)
        device = x.device

        # 1) features
        features = self.Back_bone(x)                       # [B, F]

        # 2) scores
        raw_scores = self.gate_linear(features)            # [B, E]
        if self.use_noisy_scores:
            sigma = F.softplus(self.noise_linear(features)) + self.min_noise_scale
            scores = raw_scores + sigma * torch.randn_like(raw_scores)
        else:
            scores = raw_scores

        # For logging
        gate_probs = F.softmax(raw_scores, dim=1)          # [B, E]

        # 3) balanced assignment
        if self.assign_mode == "hungarian":
            assigned_e = self._assign_hungarian(scores)    # [B]
        else:
            assigned_e = self._assign_greedy(scores)       # [B]

        # 4) efficient dispatch (top-1)
        out_dim = self.experts[0].layer2.out_features
        combined = features.new_zeros((B, out_dim))
        per_expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=device)

        # Vectorized grouping by expert: for each e, gather tokens and run the expert once
        for e in range(self.num_experts):
            token_idx = torch.nonzero(assigned_e == e, as_tuple=False).view(-1)  # [Be]
            if token_idx.numel() == 0:
                continue
            y_e = self.experts[e](features[token_idx])                           # [Be, C]
            # accumulate with index_add_ (faster than advanced indexing '+=')
            combined.index_add_(0, token_idx, y_e)
            per_expert_counts[e] = token_idx.numel()

        if return_aux:
            routing_entropy = (-gate_probs.clamp_min(1e-12).log() * gate_probs).sum(dim=1).mean()
            # In BASE there is no overflow; keep field for API compatibility
            overflow_dropped = torch.zeros(self.num_experts, dtype=torch.long, device=device)
            aux = {
                "gate_scores": raw_scores.detach(),
                "gate_probs": gate_probs.detach(),
                "topk_idx": assigned_e.view(B, 1).detach(),     # reuse field: assigned expert as 'top-1'
                "topk_scores": torch.gather(scores, 1, assigned_e.view(-1,1)).detach(),
                "topk_weights": torch.ones(B, 1, device=device),# top-1 => weight 1
                "per_expert_counts": per_expert_counts.detach(),
                "overflow_dropped": overflow_dropped.detach(),
                "capacity": self._balanced_capacities(B).sum().item(),  # equals B
                "routing_entropy": routing_entropy.detach(),
            }
            return combined, aux

        return combined

    # ----------------- train / eval -----------------

    def train_one_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                        device: torch.device, criterion: nn.Module | None = None,
                        max_batches: int | None = None):
        """
        One training epoch. BASE is strictly balanced by construction, so
        no auxiliary balancing loss is needed.
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

            outputs, _aux = self.forward(inputs, return_aux=True)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * inputs.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, device: torch.device, max_batches: int | None = None):
        """
        Evaluation; deterministic by default unless you enable noisy scores.
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
      - CIFAR-100, balanced assignment (exact if SciPy is available, else greedy).
      - Switch to CIFAR-10 by replacing expert heads and rebuilding optimizer.
    """
    print("Running basic unit test for BASE_Moe...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build BASE model
    model = BASE_Moe(
        num_experts=16,
        backbone_structure="resnet18",
        backbone_pretrained=False,
        num_features=64,      # keep in line with EC quick test
        hidden_size=32,
        output_size=100,      # CIFAR-100 first
        assign_mode="hungarian",  # will fallback to 'greedy' if SciPy not found
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
            print(f"[CIFAR-100][BASE] epoch={epoch+1} avg_loss={avg_loss:.4f} "
                  f"train_acc={acc:.4f} test_acc={test_acc:.4f}")

    # Switch to CIFAR-10 (replace expert heads and rebuild optimizer)
    print("Running basic unit test for BASE_Moe on CIFAR-10...")
    with torch.no_grad():
        for expert in model.experts:
            in_features = expert.layer2.in_features
            expert.layer2 = nn.Linear(in_features, 10).to(device)
        # out_dim read dynamically in forward

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)

    train_loader, test_loader, _ = cifar10.get_dataloaders(
        batch_size=64, num_workers=0
    )
    for epoch in range(EPOCHS):
        avg_loss, acc = model.train_one_epoch(train_loader, optimizer, device, max_batches=20)
        if (epoch + 1) % 10 == 0:
            test_acc = model.evaluate(test_loader, device, max_batches=20)
            print(f"[CIFAR-10][BASE] epoch={epoch+1} avg_loss={avg_loss:.4f} "
                  f"train_acc={acc:.4f} test_acc={test_acc:.4f}")

    print("BASE_Moe basic unit test complete.")
