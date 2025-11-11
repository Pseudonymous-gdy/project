import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Bayesian_NN_Moe (unified style, fixed F alias)
- Bayesian router only (factorized Gaussian over W,b)
- Two training modes:
    * 'expected': logistic-normal expectation (deterministic, low-variance)
    * 'mc'      : full ELBO via Monte-Carlo with optional control variate
- Sparse Top-k routing with strict truncation & renormalization
- Optional capacity per expert; efficient dispatch with index_add_
- Public API & aux dict fields are aligned with other MoE modules
"""

from typing import Optional, Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fnn   # <-- avoid name clash with local variables
from torch.utils.data import DataLoader

# project-local
import cifar10
import cifar100
from Backbone_and_Expert import Backbone, Expert


class Bayesian_NN_Moe(nn.Module):
    """
    Bayesian Router MoE with 'expected' and 'mc' ELBO modes.

    Posterior (factorized Gaussian):
        W ~ N(W_mu, diag(exp(W_logvar))) in R^{E×F}
        b ~ N(b_mu, diag(exp(b_logvar))) in R^{E}

    For features h in R^F:
        mu_m  = h @ W_mu^T + b_mu
        var_m = (h*h) @ exp(W_logvar)^T + exp(b_logvar)

    Logistic-normal expectation (per-dim shrink):
        tilde_m = mu_m / sqrt(1 + (π/8) * var_m)

    Routing & mixing:
        - Rank by logits (tilde_m for 'expected', sampled m for 'mc')
        - Top-k truncate, renormalize softmax probs strictly within Top-k
        - Capacity-aware dispatch, efficient accumulation with index_add_
    """

    def __init__(
        self,
        *,
        num_experts: int = 16,
        num_features: int = 32,
        output_size: int = 10,
        top_k: int = 4,
        # Bayesian router
        router_prior_var: float = 1.0,
        beta_kl: float = 1e-4,
        kl_anneal_steps: Optional[int] = None,
        router_temperature: float = 1.0,
        # capacity
        capacity_factor: Optional[float] = None,  # None => unlimited
        overflow_strategy: str = "drop",          # kept for API symmetry
        # training mode for data term
        router_mode: str = "expected",            # {'expected','mc'}
        elbo_samples: int = 2,                    # S >= 1 (used in 'mc')
        use_control_variate: bool = True,         # variance reduction for 'mc'
        # backbone / experts
        backbone_structure: str = "resnet18",
        backbone_pretrained: bool = False,
        hidden_size: int = 64,
        # (optional) small balance losses if you want parity with Simple_Moe
        w_importance: float = 0.0,
        w_load: float = 0.0,
    ) -> None:
        super().__init__()
        assert num_experts >= 1 and top_k >= 1
        assert router_mode in ("expected", "mc")
        assert elbo_samples >= 1

        self.num_experts = int(num_experts)
        self.num_features = int(num_features)
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.router_prior_var = float(router_prior_var)
        self.beta_kl = float(beta_kl)
        self.kl_anneal_steps = kl_anneal_steps
        self.router_temperature = float(router_temperature)
        self.capacity_factor = capacity_factor
        self.overflow_strategy = overflow_strategy

        self.router_mode = router_mode
        self.elbo_samples = int(elbo_samples)
        self.use_control_variate = bool(use_control_variate)

        self.w_importance = float(w_importance)
        self.w_load = float(w_load)

        # Deterministic backbone and experts
        self.Back_bone = Backbone(
            structure=backbone_structure,
            pretrained=backbone_pretrained,
            num_features=num_features,
        )
        self.experts = nn.ModuleList(
            [Expert(num_features=num_features, hidden_size=hidden_size, output_size=output_size)
             for _ in range(self.num_experts)]
        )

        # Router variational parameters
        E, Fdim = self.num_experts, self.num_features
        self.W_mu = nn.Parameter(torch.zeros(E, Fdim))
        self.W_logvar = nn.Parameter(torch.full((E, Fdim), -7.0))  # small var init
        self.b_mu = nn.Parameter(torch.zeros(E))
        self.b_logvar = nn.Parameter(torch.full((E,), -7.0))
        nn.init.xavier_uniform_(self.W_mu)
        nn.init.zeros_(self.b_mu)

        # step counter for KL warm-up
        self.register_buffer("_train_step", torch.tensor(0, dtype=torch.long))

    # ----------------------------- math utils -----------------------------

    def _router_moments(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (mu_m, var_m) where each is [B,E].
        mu_m  = h @ W_mu^T + b_mu
        var_m = (h*h) @ exp(W_logvar)^T + exp(b_logvar)
        """
        var_W = torch.exp(self.W_logvar)
        var_b = torch.exp(self.b_logvar)
        mu_m = Fnn.linear(h, self.W_mu, self.b_mu)                     # [B,E]
        var_m = torch.matmul(h * h, var_W.t()) + var_b.unsqueeze(0)    # [B,E]
        return mu_m, var_m.clamp_min(1e-12)

    @staticmethod
    def _logistic_normal_shrink(mu_m: torch.Tensor, var_m: torch.Tensor) -> torch.Tensor:
        """tilde_m = mu / sqrt(1 + (π/8) * var)."""
        return mu_m / torch.sqrt(1.0 + (math.pi / 8.0) * var_m)

    def _kl_router(self) -> torch.Tensor:
        """KL[q(W,b)||p(W,b)] with prior variance router_prior_var."""
        var_p = self.router_prior_var
        var_q_W = torch.exp(self.W_logvar)
        var_q_b = torch.exp(self.b_logvar)
        kl_W = 0.5 * torch.sum(
            torch.log(var_p / (var_q_W + 1e-12)) + (var_q_W + self.W_mu**2) / var_p - 1.0
        )
        kl_b = 0.5 * torch.sum(
            torch.log(var_p / (var_q_b + 1e-12)) + (var_q_b + self.b_mu**2) / var_p - 1.0
        )
        return kl_W + kl_b

    def _kl_weight(self) -> float:
        if not self.kl_anneal_steps or self.kl_anneal_steps <= 0:
            return self.beta_kl
        step = int(self._train_step.item()) + 1
        return self.beta_kl * min(1.0, step / float(self.kl_anneal_steps))

    def _compute_capacity(self, batch_size: int) -> Optional[int]:
        if self.capacity_factor is None:
            return None
        fair = (batch_size * self.top_k) / float(self.num_experts)
        cap = int(math.ceil(float(self.capacity_factor) * fair))
        return max(1, cap)

    @staticmethod
    def _cv2(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Coefficient of variation squared."""
        m = x.mean()
        v = x.var(unbiased=False)
        return v / (m * m + eps)

    # ---------------------- routing & mixing (single pass) ----------------------

    def _route_and_mix(self, h: torch.Tensor, gate_scores: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        One routing+mix pass with strict Top-k truncation & renormalized softmax.
        Args:
            h           : [B,F] features
            gate_scores : [B,E] ranking logits for Top-k (tilde_m or sampled m)
        Returns:
            combined    : [B,C] logits after mixing expert outputs
            aux dict with aligned keys (for logging/analysis)
        """
        device = h.device
        B = h.size(0)
        E = self.num_experts
        T = self.router_temperature

        # Softmax over all experts (will be truncated to Top-k below)
        gate_probs = Fnn.softmax(gate_scores / T, dim=-1)                # [B,E]

        # Top-k selection by the same gate_scores
        k = self.top_k
        topk_scores, topk_idx = torch.topk(gate_scores, k, dim=1)       # [B,k], [B,k]
        topk_weights = torch.gather(gate_probs, 1, topk_idx)             # [B,k]
        denom = topk_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        topk_weights = topk_weights / denom                              # renormalized in Top-k

        # Capacity-aware dispatch
        capacity = self._compute_capacity(B)
        per_expert_counts = torch.zeros(E, dtype=torch.long, device=device)
        overflow_dropped = torch.zeros(E, dtype=torch.long, device=device)

        out_dim = self.experts[0].layer2.out_features
        combined = h.new_zeros((B, out_dim))

        # Efficient accumulation: only run experts that appear in any Top-k
        for e in range(E):
            appear_mask = (topk_idx == e).any(dim=1)                     # [B]
            if not appear_mask.any():
                continue

            token_idx = torch.nonzero(appear_mask, as_tuple=False).view(-1)  # [Be]
            Be = token_idx.numel()

            # capacity per expert (keep tokens with largest mixing weight for this expert)
            if capacity is not None and Be > capacity:
                # compute weights for expert e among selected tokens
                w_tmp = (topk_idx[token_idx] == e).float()               # [Be,k] indicator
                w_tmp = (topk_weights[token_idx] * w_tmp).sum(dim=1)     # [Be]
                vals, order = torch.sort(w_tmp, descending=True)
                keep = order[:capacity]
                token_idx = token_idx[keep]
                overflow_dropped[e] += (Be - capacity)
                Be = capacity

            # Run expert e on its tokens
            y_e = self.experts[e](h[token_idx])                          # [Be,C]

            # Final per-token weight for expert e (use normalized Top-k probs)
            w = (topk_idx[token_idx] == e).float()                       # [Be,k]
            w = (topk_weights[token_idx] * w).sum(dim=1, keepdim=True)   # [Be,1]

            # Accumulate
            combined.index_add_(0, token_idx, w * y_e)
            per_expert_counts[e] += Be

        # Aux (aligned with other modules)
        routing_entropy = (-gate_probs.clamp_min(1e-12).log() * gate_probs).sum(dim=1).mean()
        aux = {
            "gate_scores": gate_scores.detach(),
            "gate_probs": gate_probs.detach(),
            "topk_idx": topk_idx.detach(),
            "topk_scores": topk_scores.detach(),
            "topk_weights": topk_weights.detach(),
            "per_expert_counts": per_expert_counts.detach(),
            "overflow_dropped": overflow_dropped.detach(),
            "capacity": (None if capacity is None else int(capacity)),
            "routing_entropy": routing_entropy.detach(),
        }
        return combined, aux

    # ------------------------------ forward (eval) ------------------------------

    def forward(self, x: torch.Tensor, *, return_aux: bool = False):
        """
        Deterministic evaluation path:
          - compute (mu,var), shrink to tilde_m, route with Top-k on tilde_m
          - softmax(tilde_m/T) truncated & renormalized, mix experts
        """
        h = self.Back_bone(x)                               # [B,F]
        mu_m, var_m = self._router_moments(h)               # [B,E]
        tilde_m = self._logistic_normal_shrink(mu_m, var_m) # [B,E]
        combined, aux = self._route_and_mix(h, tilde_m)

        if return_aux:
            aux_out = dict(aux)
            aux_out.update({
                "mu_logits": mu_m.detach(),
                "var_logits": var_m.detach(),
                "kl": self._kl_router(),                    # tensor (not detached) used in train
            })
            return combined, aux_out
        return combined

    # --------------------------- data terms (train) ---------------------------

    def _data_term_expected(self, h: torch.Tensor, targets: torch.Tensor,
                            criterion: nn.Module) -> Tuple[torch.Tensor, Dict]:
        """Deterministic surrogate of E_q[-log p(y|x,m)] using logistic-normal expectation."""
        mu_m, var_m = self._router_moments(h)
        tilde_m = self._logistic_normal_shrink(mu_m, var_m)
        logits, aux = self._route_and_mix(h, tilde_m)
        ce = criterion(logits, targets)
        aux.update({"mu_logits": mu_m.detach(), "var_logits": var_m.detach()})
        return ce, aux

    def _data_term_mc(self, h: torch.Tensor, targets: torch.Tensor,
                      criterion: nn.Module) -> Tuple[torch.Tensor, Dict]:
        """
        Monte-Carlo estimate of E_q[-log p(y|x,m)] with S samples.
        Optional control variate: subtract a no-grad baseline based on tilde_m.
        criterion: nn.Module that returns per-batch mean loss.
        """
        mu_m, var_m = self._router_moments(h)
        std_m = torch.sqrt(var_m)

        # Analytic baseline (no grad)
        if self.use_control_variate:
            with torch.no_grad():
                tilde_m = self._logistic_normal_shrink(mu_m, var_m)
                logits_b, _ = self._route_and_mix(h, tilde_m)
                baseline = criterion(logits_b, targets)
        else:
            baseline = None

        ces = []
        last_aux = None
        for _ in range(self.elbo_samples):
            eps = torch.randn_like(mu_m)
            m = mu_m + std_m * eps
            logits_s, aux_s = self._route_and_mix(h, m)
            ce_s = criterion(logits_s, targets)
            if baseline is not None:
                # (ce_s - baseline).detach() + baseline  -> unbiased, lower-variance
                ce_s = ce_s - baseline.detach() + baseline
            ces.append(ce_s)
            last_aux = aux_s  # keep the latest aux for logging (lightweight)
        ce = torch.stack(ces, dim=0).mean()

        aux = dict(last_aux) if last_aux is not None else {}
        aux.update({"mu_logits": mu_m.detach(), "var_logits": var_m.detach()})
        return ce, aux

    # ------------------------------ balance loss ------------------------------

    def _aux_balance_loss(self, gate_probs: torch.Tensor, topk_idx: torch.Tensor,
                          per_expert_counts: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Optional CV^2 balancing (kept for parity with Simple_Moe).
        Set w_importance=w_load=0 to disable.
        """
        if self.w_importance == 0.0 and self.w_load == 0.0:
            return torch.tensor(0.0, device=gate_probs.device)

        importance = gate_probs.sum(dim=0)                               # [E]
        loss_imp = self._cv2(importance) * self.w_importance
        if per_expert_counts is not None:
            load = per_expert_counts.to(gate_probs.dtype)
        else:
            B, E = gate_probs.shape
            onehot = torch.zeros(B, E, device=gate_probs.device, dtype=gate_probs.dtype)
            for j in range(topk_idx.size(1)):
                onehot.scatter_add_(1, topk_idx[:, j:j+1], torch.ones_like(onehot[:, :1]))
            load = onehot.sum(dim=0)
        loss_load = self._cv2(load) * self.w_load
        return loss_imp + loss_load

    # ------------------------------ train / eval ------------------------------

    def train_one_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                        device: torch.device, criterion: Optional[nn.Module] = None,
                        max_batches: Optional[int] = None):
        """
        Train one epoch:
          - compute features once
          - data term: 'expected' (deterministic) or 'mc' (ELBO with S samples)
          - add annealed KL and optional (small) balance loss
          - accuracy is computed via deterministic expected path
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.train()
        running_loss, correct, total = 0.0, 0, 0

        for bidx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and bidx >= max_batches:
                break

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # 1) features once
            h = self.Back_bone(inputs)

            # 2) data term
            if self.router_mode == "expected":
                ce, aux = self._data_term_expected(h, targets, criterion)
            else:
                ce, aux = self._data_term_mc(h, targets, criterion)

            # 3) KL (per-batch scaling) + optional small balance loss
            kl_term = self._kl_weight() * self._kl_router() / max(1, inputs.shape[0])
            bal = self._aux_balance_loss(aux["gate_probs"], aux["topk_idx"], aux["per_expert_counts"])

            loss = ce + kl_term + bal
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self._train_step.add_(1)

            # accuracy via deterministic expected path
            with torch.no_grad():
                mu_m, var_m = self._router_moments(h)
                tilde_m = self._logistic_normal_shrink(mu_m, var_m)
                logits_eval, _ = self._route_and_mix(h, tilde_m)
                preds = logits_eval.argmax(1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)
                running_loss += float(loss.item()) * inputs.size(0)

        avg_loss = running_loss / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        return avg_loss, acc

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, device: torch.device, max_batches: Optional[int] = None) -> float:
        """
        Deterministic evaluation using the logistic-normal expectation path.
        """
        self.eval()
        correct, total = 0, 0
        for bidx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and bidx >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = self.forward(inputs, return_aux=False)  # deterministic path
            preds = logits.argmax(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
        return correct / total if total > 0 else 0.0


# ------------------ Minimal quick test ------------------
if __name__ == "__main__":
    """
    (A) Randomized smoke
    (B) Tiny CIFAR sanity runs for 'expected' and 'mc'
    """
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running basic unit test for Bayesian_NN_Moe...")

    # (A) smoke
    B, C, H, W = 8, 3, 32, 32
    E, Fdim, O = 6, 32, 10  # <-- avoid shadowing functional alias
    x = torch.randn(B, C, H, W, device=device)

    model = Bayesian_NN_Moe(
        num_experts=E, num_features=Fdim, output_size=O, top_k=3,
        router_prior_var=1.0, beta_kl=1e-3, kl_anneal_steps=500,
        backbone_structure="resnet18", backbone_pretrained=False, hidden_size=64,
        capacity_factor=1.0, router_temperature=1.2,
        router_mode="expected", w_importance=0.0, w_load=0.0,
    ).to(device)

    logits, aux = model.forward(x, return_aux=True)
    assert logits.shape == (B, O)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux["var_logits"]).all()
    assert model._kl_router().item() >= 0.0
    print("[UnitTest A] expected path OK")

    # mc path one shot
    model_mc = Bayesian_NN_Moe(
        num_experts=E, num_features=Fdim, output_size=O, top_k=3,
        router_prior_var=1.0, beta_kl=1e-3, kl_anneal_steps=500,
        backbone_structure="resnet18", backbone_pretrained=False, hidden_size=64,
        capacity_factor=1.0, router_temperature=1.2,
        router_mode="mc", elbo_samples=2, w_importance=0.0, w_load=0.0,
    ).to(device)
    h = model_mc.Back_bone(x)
    ce_mc, _ = model_mc._data_term_mc(h, torch.randint(0, O, (B,), device=device), nn.CrossEntropyLoss())
    assert torch.isfinite(ce_mc)
    print("[UnitTest A] mc (ELBO) path OK")

    # (B) CIFAR sanity
    print("[UnitTest B] CIFAR-100/10 quick sanity run...")

    # CIFAR-100 expected path
    model = Bayesian_NN_Moe(
        num_experts=16, num_features=32, output_size=100, top_k=2,
        router_prior_var=10.0, beta_kl=1e-6, kl_anneal_steps=5000,
        backbone_structure="resnet18", backbone_pretrained=False, hidden_size=64,
        capacity_factor=1.5, router_temperature=1.2,
        router_mode="expected", w_importance=0.0, w_load=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
    train_loader, test_loader, _ = cifar100.get_dataloaders(batch_size=64, num_workers=0, data_dir="./data", download=False)
    for epoch in range(50):
        avg_loss, acc = model.train_one_epoch(train_loader, optimizer, device, max_batches=10)
        test_acc = model.evaluate(test_loader, device, max_batches=10)
        if (epoch + 1) % 10 == 0:
            print(f"[CIFAR-100][expected] epoch={epoch+1} avg_loss={avg_loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}")

    # CIFAR-10 expected path
    with torch.no_grad():
        for expert in model.experts:
            in_features = expert.layer2.in_features
            expert.layer2 = nn.Linear(in_features, 10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
    train_loader, test_loader, _ = cifar10.get_dataloaders(batch_size=64, num_workers=0, data_dir="./data", download=False)
    for epoch in range(50):
        avg_loss, acc = model.train_one_epoch(train_loader, optimizer, device, max_batches=10)
        test_acc = model.evaluate(test_loader, device, max_batches=10)
        if (epoch + 1) % 10 == 0:
            print(f"[CIFAR-10][expected] epoch={epoch+1} avg_loss={avg_loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}")
    # # CIFAR-10 mc (ELBO)
    # model_mc = Bayesian_NN_Moe(
    #     num_experts=16, num_features=32, output_size=10, top_k=4,
    #     router_prior_var=1.0, beta_kl=1e-4, kl_anneal_steps=500,
    #     backbone_structure="resnet18", backbone_pretrained=False, hidden_size=64,
    #     capacity_factor=1.0, router_temperature=1.2,
    #     router_mode="mc", elbo_samples=4, w_importance=1e-3, w_load=0.0,
    # ).to(device)
    # optimizer = torch.optim.AdamW(model_mc.parameters(), lr=1e-4, weight_decay=1e-4)
    # for epoch in range(50):
    #     avg_loss, acc = model_mc.train_one_epoch(train_loader, optimizer, device, max_batches=10)
    #     test_acc = model_mc.evaluate(test_loader, device, max_batches=10)
    #     if (epoch + 1) % 10 == 0:
    #         print(f"[CIFAR-10][mc] epoch={epoch+1} avg_loss={avg_loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}")

    # print("Bayesian_NN_Moe basic unit test complete.")
