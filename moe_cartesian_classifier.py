import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        # use SwiGLU: project to 2 * d_ff, split into (x, gate), apply SiLU to gate and multiply
        self.w1 = nn.Linear(d_model, d_ff * 2)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w1(x)
        a, b = x.chunk(2, dim=-1)
        # SwiGLU: a * SiLU(b)
        return self.w2(self.dropout(a * F.silu(b)))

class CartesianMoE(nn.Module):
    def __init__(self, d_model, d_ff, n_axisA, n_axisB, dropout=0.0):
        super().__init__()
        self.n_axisA = n_axisA
        self.n_axisB = n_axisB
        # experts as [axisA][axisB]
        self.experts = nn.ModuleList([
            nn.ModuleList([FeedForward(d_model, d_ff, dropout) for _ in range(n_axisB)])
            for _ in range(n_axisA)
        ])
        self.gate_A = nn.Linear(d_model, n_axisA)
        self.gate_B = nn.Linear(d_model, n_axisB)

    def forward(self, x):
        b, s, d = x.shape
        x_flat = x.view(b*s, d)
        # gate probabilities
        probs_A = F.softmax(self.gate_A(x_flat), dim=-1)
        probs_B = F.softmax(self.gate_B(x_flat), dim=-1)
        idx_A = torch.argmax(probs_A, dim=-1)
        idx_B = torch.argmax(probs_B, dim=-1)

        # aux loss for load balancing
        importance_A = probs_A.mean(dim=0)
        one_hot_A = F.one_hot(idx_A, num_classes=self.n_axisA).float()
        load_A = one_hot_A.mean(dim=0)
        aux_A = (self.n_axisA * (importance_A * load_A).sum())

        importance_B = probs_B.mean(dim=0)
        one_hot_B = F.one_hot(idx_B, num_classes=self.n_axisB).float()
        load_B = one_hot_B.mean(dim=0)
        aux_B = (self.n_axisB * (importance_B * load_B).sum())

        total_aux = aux_A + aux_B

        # dispatch tokens to experts
        outputs = torch.zeros_like(x_flat)
        for i in range(self.n_axisA):
            mask_i = (idx_A == i)
            if not mask_i.any():
                continue
            for j in range(self.n_axisB):
                mask_j = (idx_B == j)
                mask = mask_i & mask_j
                if not mask.any():
                    continue
                outputs[mask] = self.experts[i][j](x_flat[mask])

        return outputs.view(b, s, d), total_aux


# -------------------------
# Transformer block with optional Cartesian MoE
# -------------------------
class TransformerEncoderLayerCartesian(nn.Module):
    def __init__(self, d_model, nhead, d_ff, n_axisA=None, n_axisB=None, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        if n_axisA is None or n_axisB is None:
            self.ffn = FeedForward(d_model, d_ff, dropout)
            self.is_moe = False
        else:
            self.ffn = CartesianMoE(d_model, d_ff, n_axisA, n_axisB, dropout)
            self.is_moe = True

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)
        if self.is_moe:
            ffn_out, aux = self.ffn(x)
            x = x + self.dropout(ffn_out)
            x = self.ln2(x)
            return x, aux
        else:
            x = x + self.dropout(self.ffn(x))
            x = self.ln2(x)
            return x, torch.tensor(0.0, device=x.device)


# -------------------------
# Full Cartesian MoE classifier
# -------------------------
class CartesianMoEGenreClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, num_layers=6,
                 d_ff=2048, n_axisA=2, n_axisB=2, moe_layers=None,
                 seq_len=256, num_classes=10, dropout=0.1, use_cls_token=False):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.use_cls_token = use_cls_token
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len + (1 if use_cls_token else 0), d_model) * 0.01)

        self.layers = nn.ModuleList()
        moe_set = set(moe_layers or [])
        for i in range(num_layers):
            if i in moe_set:
                self.layers.append(TransformerEncoderLayerCartesian(d_model, n_heads, d_ff, n_axisA, n_axisB, dropout))
            else:
                self.layers.append(TransformerEncoderLayerCartesian(d_model, n_heads, d_ff, None, None, dropout))

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1,1,d_model))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, 0.0, 0.02)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask=None):
        b, seq = input_ids.shape
        device = input_ids.device
        x = self.token_emb(input_ids)
        if self.use_cls_token:
            cls = self.cls_token.expand(b,-1,-1)
            x = torch.cat([cls, x], dim=1)
        x = x + self.pos_emb[:, :x.size(1), :]

        total_aux = torch.tensor(0.0, device=device)
        for layer in self.layers:
            x, aux = layer(x)
            total_aux = total_aux + aux

        if self.use_cls_token:
            rep = x[:,0,:]
        else:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1)
                rep = (x*mask).sum(dim=1)/(mask.sum(dim=1).clamp(min=1e-6))
            else:
                rep = x.mean(dim=1)

        logits = self.classifier(rep)
        return logits, total_aux


# -------------------------
# Fake dataset for testing
# -------------------------
class FakeLyricsDataset(Dataset):
    def __init__(self, vocab_size, seq_len, num_samples=1024, num_classes=10):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        label = torch.randint(0, self.num_classes, (1,), dtype=torch.long).item()
        attention = torch.ones(self.seq_len, dtype=torch.long)
        return tokens, attention, label


# -------------------------
# Training with AMP + scheduler
# -------------------------
def train_cartesian_amp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 30_000
    seq_len = 256
    num_classes = 12
    d_model = 512
    d_ff = 2048
    num_layers = 6
    n_axisA, n_axisB = 2, 2
    moe_layers = [2,4]
    batch_size = 24
    num_epochs = 2
    aux_loss_weight = 0.01
    max_grad_norm = 1.0

    model = CartesianMoEGenreClassifier(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=8,
        num_layers=num_layers,
        d_ff=d_ff,
        n_axisA=n_axisA,
        n_axisB=n_axisB,
        moe_layers=moe_layers,
        seq_len=seq_len,
        num_classes=num_classes,
        dropout=0.1,
        use_cls_token=False,
    ).to(device)

    dataset = FakeLyricsDataset(vocab_size, seq_len, num_samples=2000, num_classes=num_classes)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()

    total_steps = len(loader) * num_epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lr_lambda=lambda step: min((step+1)/warmup_steps,1.0)*0.5*(1+math.cos(math.pi*step/total_steps))
    )

    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(num_epochs):
        for it, (tokens, attention, labels) in enumerate(loader):
            tokens, attention, labels = tokens.to(device), attention.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits, aux = model(tokens, attention_mask=attention)
                loss_main = ce(logits, labels)
                loss = loss_main + aux_loss_weight * aux
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if it % 20 == 0:
                acc = (logits.argmax(-1) == labels).float().mean().item()
                print(f"Epoch {epoch} Iter {it} Loss={loss.item():.4f} Main={loss_main.item():.4f} Aux={aux.item():.4f} Acc={acc:.3f}")

    print("Cartesian MoE training with AMP complete.")


if __name__ == "__main__":
    train_cartesian_amp()