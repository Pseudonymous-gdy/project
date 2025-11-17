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

# Small helper for quick CIFAR sanity tests at the bottom
MAX_BATCHES_QTEST = 10


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

    Warm-up (BASE-style but Bayesian):
        - Sample router parameters via reparameterization.
        - Use sampled gate scores + balanced greedy Top-k assignment.
        - Train only on data loss (no KL) to obtain a good, balanced prior.
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
        """
        Linear KL warm-up:
          beta_kl * min(1, t / kl_anneal_steps)
        """
        if not self.kl_anneal_steps or self.kl_anneal_steps <= 0:
            return self.beta_kl
        step = int(self._train_step.item()) + 1
        return self.beta_kl * min(1.0, step / float(self.kl_anneal_steps))

    def _compute_capacity(self, batch_size: int) -> Optional[int]:
        """
        Compute per-expert capacity used in the main Top-k router.

        capacity_factor = 1.0  -> 'fair share'
        capacity_factor > 1.0  -> allow overflow
        """
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

    # ---------------- Bayesian sampling used in warm-up -----------------

    def _sample_router_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterization sampling of router weights:

            W = W_mu + exp(0.5 * W_logvar) * eps_W
            b = b_mu + exp(0.5 * b_logvar) * eps_b

        This is only used in the warm-up phase, where we treat the sampled
        parameters as if they were a deterministic BASE-style gate and
        train *without* KL, purely on data loss.
        """
        std_W = torch.exp(0.5 * self.W_logvar)
        std_b = torch.exp(0.5 * self.b_logvar)
        eps_W = torch.randn_like(self.W_mu)
        eps_b = torch.randn_like(self.b_mu)
        W = self.W_mu + std_W * eps_W
        b = self.b_mu + std_b * eps_b
        return W, b

    # ---------------------- routing & mixing (single pass) ----------------------

    def _route_and_mix(self, h: torch.Tensor, gate_scores: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        One routing+mix pass with strict Top-k truncation & renormalized softmax.

        Args:
            h           : [B,F] features from the backbone
            gate_scores : [B,E] ranking logits for Top-k (tilde_m or sampled m)

        Returns:
            combined    : [B,C] logits after mixing expert outputs
            aux dict    : diagnostics aligned with other MoE modules
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
          - softmax(tilde_m/T) truncated & renormalized, mix experts.
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
        """
        Deterministic surrogate of E_q[-log p(y|x,m)] using logistic-normal expectation.
        """
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
        Train one epoch on the full Bayesian-ELBO objective:

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

    # ------------------------------------------------------------------
    # BASE-style warm-up path: balanced Top-k assignment with Bayesian gate
    # ------------------------------------------------------------------

    def _warmup_balanced_capacities(self, total_slots: int) -> torch.Tensor:
        """
        Balanced capacities per expert for a *total* number of slots.

        Here, "slot" means a (token, expert) assignment. For Top-k warm-up,
        each token contributes k slots, so total_slots = B * k_warmup.

        We return integer capacities per expert such that:
          - capacities.sum() == total_slots
          - capacities differ by at most 1
        """
        E = self.num_experts
        q, r = divmod(total_slots, E)
        caps = torch.full((E,), q, dtype=torch.long)
        if r > 0:
            caps[:r] += 1
        return caps  # [E]

    def _warmup_assign_greedy_topk(self, scores: torch.Tensor, k_warmup: int) -> torch.Tensor:
        """
        Greedy *balanced* Top-k assignment used in warm-up.

        Args:
          scores   : [B, E] token->expert scores (higher is better).
          k_warmup : number of experts per token in warm-up (k_warmup >= 1).

        Returns:
          topk_idx : [B, k_warmup] long tensor.
                     For each token b, topk_idx[b] lists the experts assigned
                     to that token. Each token appears in exactly k_warmup
                     (token, expert) pairs, and expert loads are globally
                     balanced as much as possible.
        """
        device = scores.device
        B, E = scores.shape

        # Make sure k_warmup is feasible
        k_warmup = max(1, min(k_warmup, E))

        # Total number of (token, expert) assignments (slots) we want
        total_slots = B * k_warmup

        # Balanced per-expert capacities; sum(capacities) == total_slots
        caps = self._warmup_balanced_capacities(total_slots).to(device)  # [E]

        # token_deg[b]: how many experts token b has been assigned to so far
        token_deg = torch.zeros(B, dtype=torch.long, device=device)

        # assigned[b, e] = True if token b is assigned to expert e
        assigned = torch.zeros(B, E, dtype=torch.bool, device=device)

        # Flatten and sort all (token, expert) pairs by score desc
        flat = scores.reshape(-1)                        # [B*E]
        order = torch.argsort(flat, descending=True)     # indices into flat

        remaining = total_slots
        for idx in order:
            if remaining == 0:
                break
            b = (idx // E).item()
            e = (idx % E).item()
            # Token already has k_warmup experts?
            if token_deg[b] >= k_warmup:
                continue
            # Expert at capacity?
            if caps[e] <= 0:
                continue
            # Already assigned this (b,e)?
            if assigned[b, e]:
                continue
            # Commit this assignment
            assigned[b, e] = True
            token_deg[b] += 1
            caps[e] -= 1
            remaining -= 1

        # First fallback: try to use remaining expert capacity (if any) to fill
        # tokens that have fewer than k_warmup assignments.
        need_mask = token_deg < k_warmup
        if need_mask.any():
            for b in range(B):
                need = k_warmup - token_deg[b].item()
                if need <= 0:
                    continue
                scores_b = scores[b]                      # [E]
                order_e = torch.argsort(scores_b, descending=True)
                for e in order_e.tolist():
                    if need <= 0:
                        break
                    if assigned[b, e]:
                        continue
                    if caps[e] <= 0:
                        continue
                    assigned[b, e] = True
                    token_deg[b] += 1
                    caps[e] -= 1
                    need -= 1

        # Second fallback: if some tokens still have < k_warmup assignments,
        # relax capacities and purely complete by local Top-k.
        need_mask = token_deg < k_warmup
        if need_mask.any():
            for b in range(B):
                need = k_warmup - token_deg[b].item()
                if need <= 0:
                    continue
                scores_b = scores[b]
                order_e = torch.argsort(scores_b, descending=True)
                for e in order_e.tolist():
                    if need <= 0:
                        break
                    if assigned[b, e]:
                        continue
                    assigned[b, e] = True
                    token_deg[b] += 1
                    need -= 1

        # Build topk_idx per token from the assigned matrix.
        topk_idx = torch.empty(B, k_warmup, dtype=torch.long, device=device)
        for b in range(B):
            experts_b = torch.nonzero(assigned[b], as_tuple=False).view(-1)  # [deg_b]
            if experts_b.numel() > k_warmup:
                # If more than k_warmup, keep only the best-scoring ones
                scores_b = scores[b, experts_b]
                order_e = torch.argsort(scores_b, descending=True)
                experts_b = experts_b[order_e[:k_warmup]]
            elif experts_b.numel() < k_warmup:
                # Should be rare due to the second fallback, but guard anyway:
                # pad with best-scoring experts not yet chosen for token b.
                scores_b_all = scores[b]
                order_all = torch.argsort(scores_b_all, descending=True)
                for e in order_all.tolist():
                    if experts_b.numel() >= k_warmup:
                        break
                    if (experts_b == e).any():
                        continue
                    experts_b = torch.cat(
                        [experts_b, torch.tensor([e], dtype=torch.long, device=device)],
                        dim=0,
                    )
            topk_idx[b] = experts_b[:k_warmup]

        return topk_idx  # [B, k_warmup]

    def forward_warmup(self, x: torch.Tensor, *, return_aux: bool = False):
        """
        Warm-up forward (Bayesian BASE-style Top-k):

          1) Sample router parameters W,b via reparameterization.
          2) Compute gate_scores = h @ W^T + b for each token.
          3) Run a *balanced* greedy Top-k assignment on these scores:
               - Each token is assigned to k_warmup experts.
               - Total expert loads are globally balanced as much as possible.
          4) Mix expert outputs using normalized gate probabilities restricted
             to the assigned Top-k.

        Differences from the main path:
          - No KL term is considered here.
          - Capacity factor is encoded via the balanced assignment itself.
          - Objective during warm-up is *pure* data loss (CE), used only to:
              * stabilize expert specialization
              * provide a good empirical prior for router parameters.
        """
        device = x.device
        B = x.size(0)

        # 1) features
        h = self.Back_bone(x)  # [B, F]

        # 2) sample router parameters and compute scores
        W_sample, b_sample = self._sample_router_params()
        gate_scores = Fnn.linear(h, W_sample, b_sample)  # [B, E]

        # Softmax over experts for this sample; we still use temperature here
        gate_probs = Fnn.softmax(gate_scores / self.router_temperature, dim=1)  # [B, E]

        # 3) balanced Top-k assignment (greedy, BASE-style but Bayesian scores)
        #    Here we recommend k_warmup = self.top_k to keep the sparsity pattern
        #    compatible with the main training regime. If you want a slightly denser
        #    warm-up (more overlap between experts), you may try:
        #       k_warmup = int(math.ceil(1.5 * self.top_k))
        k_warmup = max(1, min(self.top_k, self.num_experts))
        topk_idx = self._warmup_assign_greedy_topk(gate_scores, k_warmup)  # [B, k_warmup]

        # 4) use normalized probabilities within the assigned Top-k set
        topk_weights = torch.gather(gate_probs, 1, topk_idx)              # [B, k_warmup]
        denom = topk_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        topk_weights = topk_weights / denom

        out_dim = self.experts[0].layer2.out_features
        combined = h.new_zeros((B, out_dim))
        per_expert_counts = torch.zeros(self.num_experts, dtype=torch.long, device=device)

        # Dispatch & mix, mirroring the main _route_and_mix but without overflow logic
        for e in range(self.num_experts):
            appear_mask = (topk_idx == e).any(dim=1)                      # [B]
            if not appear_mask.any():
                continue
            token_idx = torch.nonzero(appear_mask, as_tuple=False).view(-1)  # [Be]
            Be = token_idx.numel()

            y_e = self.experts[e](h[token_idx])                           # [Be, C]

            # Final per-token weight for expert e (normalized within its Top-k)
            w = (topk_idx[token_idx] == e).float()                        # [Be, k_warmup]
            w = (topk_weights[token_idx] * w).sum(dim=1, keepdim=True)    # [Be, 1]

            combined.index_add_(0, token_idx, w * y_e)
            per_expert_counts[e] = Be

        if not return_aux:
            return combined

        # logging fields kept API-compatible with other MoE modules
        routing_entropy = (-gate_probs.clamp_min(1e-12).log() * gate_probs).sum(dim=1).mean()
        overflow_dropped = torch.zeros(self.num_experts, dtype=torch.long, device=device)
        total_slots = B * k_warmup
        capacity = int(self._warmup_balanced_capacities(total_slots).sum().item())  # == total_slots

        aux = {
            "gate_scores": gate_scores.detach(),
            "gate_probs": gate_probs.detach(),
            "topk_idx": topk_idx.detach(),
            "topk_scores": torch.gather(gate_scores, 1, topk_idx).detach(),
            "topk_weights": topk_weights.detach(),
            "per_expert_counts": per_expert_counts.detach(),
            "overflow_dropped": overflow_dropped.detach(),
            "capacity": capacity,
            "routing_entropy": routing_entropy.detach(),
        }
        return combined, aux

    def train_one_epoch_warmup(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        criterion: Optional[nn.Module] = None,
        max_batches: Optional[int] = None,
    ):
        """
        Warm-up training epoch (Bayesian + BASE-style Top-k inside the same model):

          - Uses forward_warmup (balanced Top-k assignment from *sampled* router).
          - Only optimizes CE loss (no KL, no balance loss).
          - Does NOT touch self._train_step (so KL annealing for real training
            starts fresh after warm-up).

        Goal:
          - Get a reasonable, load-balanced *router prior* and a good initial
            expert specialization before turning on the full ELBO.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        self.train()
        running_loss, correct, total = 0, 0, 0

        for bidx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and bidx >= max_batches:
                break

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            outputs, _aux = self.forward_warmup(inputs, return_aux=True)
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
    def evaluate_warmup(
        self,
        loader: DataLoader,
        device: torch.device,
        max_batches: Optional[int] = None,
    ) -> float:
        """
        Evaluation under the warm-up routing scheme (forward_warmup).
        """
        self.eval()
        correct, total = 0, 0
        for bidx, (inputs, targets) in enumerate(loader):
            if max_batches is not None and bidx >= max_batches:
                break

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = self.forward_warmup(inputs, return_aux=False)
            preds = logits.argmax(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

        return correct / total if total > 0 else 0.0


# ------------------ Minimal quick test ------------------
if __name__ == "__main__":
    """
    (A) Randomized smoke
    (B) Tiny CIFAR sanity runs for warm-up + 'expected' mode
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

    # check main deterministic forward
    logits, aux = model.forward(x, return_aux=True)
    assert logits.shape == (B, O)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(aux["var_logits"]).all()
    assert model._kl_router().item() >= 0.0
    print("[UnitTest A] expected path OK")

    # check warm-up forward (Bayesian + balanced Top-k)
    logits_warm, aux_warm = model.forward_warmup(x, return_aux=True)
    assert logits_warm.shape == (B, O)
    assert torch.isfinite(logits_warm).all()
    assert aux_warm["topk_idx"].shape[1] == model.top_k
    print("[UnitTest A] warm-up path OK")

    # (B) CIFAR sanity
    print("[UnitTest B] CIFAR-100/10 quick sanity run...")

    # CIFAR-100: warm-up then expected path
    model = Bayesian_NN_Moe(
        num_experts=16, num_features=32, output_size=100, top_k=2,
        router_prior_var=10.0, beta_kl=1e-6, kl_anneal_steps=5000,
        backbone_structure="resnet18", backbone_pretrained=False, hidden_size=64,
        capacity_factor=1.5, router_temperature=1.2,
        router_mode="expected", w_importance=0.0, w_load=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)

    train_loader, test_loader, _ = cifar100.get_dataloaders(
        batch_size=64, num_workers=0, data_dir="./data", download=False
    )

    # Warm-up for a few epochs
    for epoch in range(20):
        avg_loss, acc = model.train_one_epoch_warmup(
            train_loader, optimizer, device, max_batches=MAX_BATCHES_QTEST
        )
        val_acc = model.evaluate_warmup(
            test_loader, device, max_batches=MAX_BATCHES_QTEST
        )
        if (epoch + 1) % 10 == 0:
            print(f"[warmup] epoch={epoch+1} "
                  f"loss={avg_loss:.4f} train_acc={acc:.4f} val_acc={val_acc:.4f}")

    # Then switch to full Bayesian training (expected path)
    for epoch in range(30):
        avg_loss, acc = model.train_one_epoch(
            train_loader, optimizer, device, max_batches=MAX_BATCHES_QTEST
        )
        test_acc = model.evaluate(test_loader, device, max_batches=MAX_BATCHES_QTEST)
        if (epoch + 1) % 10 == 0:
            print(f"[CIFAR-100][expected] epoch={epoch+1} "
                  f"avg_loss={avg_loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}")

    # CIFAR-10 expected path quick check
    with torch.no_grad():
        for expert in model.experts:
            in_features = expert.layer2.in_features
            expert.layer2 = nn.Linear(in_features, 10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)
    train_loader, test_loader, _ = cifar10.get_dataloaders(
        batch_size=64, num_workers=0, data_dir="./data", download=False
    )
    for epoch in range(3):
        avg_loss, acc = model.train_one_epoch(
            train_loader, optimizer, device, max_batches=MAX_BATCHES_QTEST
        )
        test_acc = model.evaluate(test_loader, device, max_batches=MAX_BATCHES_QTEST)
        print(f"[CIFAR-10][expected] epoch={epoch+1} "
              f"avg_loss={avg_loss:.4f} train_acc={acc:.4f} test_acc={test_acc:.4f}")

    print("Bayesian_NN_Moe basic unit test complete.")
