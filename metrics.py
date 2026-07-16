"""The eight per-token uncertainty signals in Shichengf/entropy-acc-traces.

All signals are computed from the RAW logits (temperature 1.0), i.e. the model's
own next-token distribution -- not the tempered/top-p-filtered distribution used
for sampling. Entropy of a top-p-filtered distribution measures the sampler as
much as the model, so the two are kept separate: sampling params affect WHICH
token is drawn, never the recorded signals.

Definitions (natural log throughout; V = full vocab):
  entropy        H(p)  = -sum_v p_v log p_v
  varentropy     Var_p[-log p] = sum_v p_v (-log p_v - H)^2
  top1_prob      max_v p_v
  top2_prob      second largest p_v
  margin         top1_prob - top2_prob
  self_certainty KL(U || p) = -log|V| - (1/|V|) sum_v log p_v
                 (Zhao et al. 2025, "Scalable Best-of-N Selection ... Self-Certainty")
  p_tail         1 - sum_{v in top20} p_v
  top20_*        raw top-20 token ids and logprobs
"""

import torch

TOPK = 20


@torch.no_grad()
def step_signals(logits: torch.Tensor) -> dict:
    """Compute the eight signals for one decode step.

    Args:
        logits: (batch, vocab) raw next-token logits, unscaled.
    Returns:
        dict of per-sequence tensors, each (batch,) except top20_* which are
        (batch, 20). Computed in float32 for numerical stability.
    """
    logits = logits.float()
    logp = torch.log_softmax(logits, dim=-1)          # (B, V)
    p = logp.exp()

    # entropy / varentropy. p*logp -> 0 as p -> 0, and log_softmax never
    # returns -inf for finite logits, so no nan-guard is needed here.
    neg_logp = -logp
    H = (p * neg_logp).sum(-1)                        # (B,)
    varent = (p * (neg_logp - H.unsqueeze(-1)).pow(2)).sum(-1)

    # KL(U || p): uniform reference over the full vocab.
    V = logp.shape[-1]
    self_cert = -torch.log(torch.tensor(float(V), device=logp.device)) - logp.mean(-1)

    top = logp.topk(TOPK, dim=-1)                     # values sorted desc
    top_logp, top_ids = top.values, top.indices
    top_p = top_logp.exp()

    top1 = top_p[:, 0]
    top2 = top_p[:, 1]
    p_tail = (1.0 - top_p.sum(-1)).clamp_min(0.0)     # clamp fp error at ~0

    return {
        "entropy": H,
        "varentropy": varent,
        "top1_prob": top1,
        "top2_prob": top2,
        "margin": top1 - top2,
        "self_certainty": self_cert,
        "p_tail": p_tail,
        "top20_token_ids": top_ids,
        "top20_logprobs": top_logp,
    }


PER_STEP_KEYS = [
    "entropy", "varentropy", "top1_prob", "top2_prob", "margin",
    "self_certainty", "p_tail", "top20_token_ids", "top20_logprobs",
]
