# This is a generic Mixture of Experts (MoE) classifier model implemented in PyTorch

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Basic building blocks
# -------------------------
class FeedForward(nn.Module):
    """Simple FFN used as an expert (SwiGLU activation)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        raise NotImplementedError("Need SwiGLU impl here.")

class SwitchMoE(nn.Module):
    """
    Switch (Top-1) MoE layer.

    - d_model: hidden dim
    - d_ff: expert hidden dim
    - n_experts: number of experts
    - capacity_factor: how many tokens per expert capacity relative to average
    """
    def __init__(self, d_model: int, d_ff: int, n_experts: int, capacity_factor: float = 1.25, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.capacity_factor = capacity_factor

        # Experts: module list of independent FFNs
        self.experts = nn.ModuleList([FeedForward(d_model, d_ff, dropout=dropout) for _ in range(n_experts)])

        # Simple gating network (linear projection -> softmax)
        self.gate = nn.Linear(d_model, n_experts)

    def forward(self, x):
        """
        x: [batch, seq, d_model]
        returns: (out: [batch, seq, d_model], loss_aux: scalar)
        """
        b, s, d = x.shape
        tokens = b * s
        x_flat = x.view(tokens, d)  # [T, d]

        # gating logits & softmax
        logits = self.gate(x_flat)                 # [T, n_experts]
        probs = F.softmax(logits, dim=-1)          # [T, n_experts]

        # Top-1 selection
        top1_idx = torch.argmax(probs, dim=-1)     # [T]

        # Compute auxiliary load-balancing loss:
        # importance = average gate probability per expert
        importance = probs.mean(dim=0)             # [n_experts], average prob mass per expert
        # load = fraction of tokens routed to each expert (one-hot avg)
        one_hot = F.one_hot(top1_idx, num_classes=self.n_experts).float()  # [T, n_experts]
        load = one_hot.mean(dim=0)                 # [n_experts]

        # Switch/GShard style auxiliary loss (encourages experts to be used)
        loss_aux = (self.n_experts * (importance * load).sum())

        # Prepare outputs placeholder
        outputs = torch.zeros_like(x_flat)

        # Dispatch tokens to experts and compute expert outputs
        # NOTE: this loop is fine for small n_experts and single-GPU prototyping.
        for i, expert in enumerate(self.experts):
            mask = (top1_idx == i)
            if not mask.any():
                continue
            # select tokens for this expert
            inp = x_flat[mask]                    # [num_tokens_to_this_expert, d]
            out_i = expert(inp)                   # [num_tokens_to_this_expert, d]
            outputs[mask] = out_i

        out = outputs.view(b, s, d)
        return out, loss_aux

class TransformerEncoderLayerMoE(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_ff: int, n_experts: int = None, dropout: float = 0.1):
        """
        if n_experts is None -> use dense FFN.
        if n_experts is int -> use SwitchMoE with that many experts.
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        if n_experts is None:
            self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
            self.is_moe = False
        else:
            self.ffn = SwitchMoE(d_model, d_ff, n_experts, dropout=dropout)
            self.is_moe = True

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        x: [batch, seq, d_model]
        returns: (out, aux_loss) aux_loss is 0.0 if not MoE
        """
        # Self-attention (residual + norm)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)

        # FFN or MoE
        if self.is_moe:
            ffn_out, aux_loss = self.ffn(x)       # MoE returns (out, aux)
            x = x + self.dropout(ffn_out)
            x = self.ln2(x)
            return x, aux_loss
        else:
            ffn_out = self.ffn(x)
            x = x + self.dropout(ffn_out)
            x = self.ln2(x)
            return x, torch.tensor(0.0, device=x.device)


# -------------------------
# Full model: embedding -> encoder -> pooling -> classifier
# -------------------------
class MoEGenreClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_heads: int = 12,
        num_layers: int = 6,
        d_ff: int = 3072,
        n_experts: int = 8,
        moe_layers: list = None,
        seq_len: int = 512,
        num_classes: int = 10,
        dropout: float = 0.1,
        use_cls_token: bool = False
    ):
        """
        - moe_layers: list of layer indices (0-based) where MoE should be used.
                      e.g., [2, 4] to use MoE in layers 3 and 5.
        - use_cls_token: whether to prepend a CLS token (if True, seq_len should account for it)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.use_cls_token = use_cls_token

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len + (1 if use_cls_token else 0), d_model) * 0.01)

        self.layers = nn.ModuleList()
        moe_set = set(moe_layers or [])
        for i in range(num_layers):
            if i in moe_set:
                layer = TransformerEncoderLayerMoE(d_model, n_heads, d_ff, n_experts=n_experts, dropout=dropout)
            else:
                layer = TransformerEncoderLayerMoE(d_model, n_heads, d_ff, n_experts=None, dropout=dropout)
            self.layers.append(layer)

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        # optionally a learned CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # weight init
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [batch, seq]
        attention_mask: optional [batch, seq] with 1 for real tokens, 0 for padding
        returns: logits [batch, num_classes], aux_loss scalar
        """
        b, seq = input_ids.shape
        device = input_ids.device

        # Embeddings
        x = self.token_emb(input_ids)  # [b, seq, d_model]

        # optional CLS token
        if self.use_cls_token:
            cls = self.cls_token.expand(b, -1, -1)   # [b,1,d]
            x = torch.cat([cls, x], dim=1)           # [b, seq+1, d]
        # add pos emb (ensure pos_emb large enough)
        x = x + self.pos_emb[:, : x.size(1), :]

        # optional attention mask -> convert to attn_mask format expected by MultiheadAttention
        attn_mask = None
        if attention_mask is not None:
            # MultiheadAttention uses key_padding_mask boolean of shape [batch, seq].
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None

        total_aux = torch.tensor(0.0, device=device)
        # Encoder stack
        for layer in self.layers:
            x, aux = layer(x, attn_mask=None)
            total_aux = total_aux + aux

        # Pooling: CLS or mean-pool over non-padded tokens
        if self.use_cls_token:
            rep = x[:, 0, :]  # use CLS
        else:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)     # [b, seq, 1]
                rep = (x * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-6))
            else:
                rep = x.mean(dim=1)

        logits = self.classifier(rep)  # [b, num_classes]
        return logits, total_aux