# Transformer for predicting (job, machine, worker_group) tokens
# Includes padding + attention masking

import numpy as np
import torch
torch.set_num_threads(10)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import math
import os


# ---------------- Dataset ----------------

class EventSequenceDataset(Dataset):
    def __init__(self, sequences_file):
        """
        sequences_file: npy file with shape (N, L, 3), dtype=object
                        each entry is [job, machine, group]
        """
        self.sequences = np.load(sequences_file, allow_pickle=True)

        # ------------------------------------------------------------------
        # Build vocabulary safely from object arrays
        # ------------------------------------------------------------------
        token_set = set()
        for seq in self.sequences:
            for tok in seq:
                token_set.add(tuple(int(x) for x in tok))

        self.idx_to_token = list(sorted(token_set))
        self.token_to_idx = {tok: i for i, tok in enumerate(self.idx_to_token)}
        self.vocab_size = len(self.idx_to_token)

        # ------------------------------------------------------------------
        # Encode sequences as token indices
        # ------------------------------------------------------------------
        self.encoded_sequences = []
        for seq in self.sequences:
            enc = [self.token_to_idx[tuple(int(x) for x in tok)] for tok in seq]
            self.encoded_sequences.append(enc)

        # ------------------------------------------------------------------
        # Build training samples: prefix → next token
        # ------------------------------------------------------------------
        self.samples = []
        for seq in self.encoded_sequences:
            for t in range(1, len(seq)):
                self.samples.append((
                    torch.tensor(seq[:t], dtype=torch.long),
                    torch.tensor(seq[t], dtype=torch.long)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



# ---------------- Relative Position Bias ----------------

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_len=64):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        self.relative_bias = nn.Embedding(2 * max_len - 1, num_heads)

    def forward(self, qlen, klen, device):
        pos = torch.arange(qlen, device=device)[:, None] - torch.arange(klen, device=device)[None, :]
        pos = pos + self.max_len - 1
        pos = pos.clamp(0, 2 * self.max_len - 2)
        values = self.relative_bias(pos)
        return values.permute(2, 0, 1)  # (h, q, k)


# ---------------- Attention ----------------

class MultiheadAttentionWithRelPos(nn.Module):
    def __init__(self, d_model, nhead, max_len=64, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_head = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.rel_pos_bias = RelativePositionBias(nhead, max_len)

    def forward(self, x, mask=None):
        B, L, _ = x.shape

        Q = self.q_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.nhead, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + self.rel_pos_bias(L, L, x.device).unsqueeze(0)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


# ---------------- Encoder Layer ----------------

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff=256, dropout=0.1, max_len=64):
        super().__init__()
        self.attn = MultiheadAttentionWithRelPos(d_model, nhead, max_len, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ff(x))
        return x


# ---------------- Transformer ----------------

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4, max_len=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, max_len=max_len)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.out(x[:, -1])


# ---------------- Collate ----------------

def collate_fn(batch):
    prefixes, targets = zip(*batch)
    max_len = max(len(p) for p in prefixes)

    padded = torch.zeros(len(prefixes), max_len, dtype=torch.long)
    mask = torch.zeros(len(prefixes), max_len)

    for i, p in enumerate(prefixes):
        padded[i, :len(p)] = p
        mask[i, :len(p)] = 1

    return padded, torch.stack(targets), mask


# ---------------- Training ----------------

def train_model(sequences_file, epochs=300, batch_size=32, lr=1e-4, device="cpu"):
    dataset = EventSequenceDataset(sequences_file)

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=collate_fn)

    model = TransformerModel(dataset.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y, mask in train_loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            logits = model(x, mask)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y, mask in val_loader:
                logits = model(x.to(device), mask.to(device))
                pred = logits.argmax(dim=-1)
                correct += (pred.cpu() == y).sum().item()
                total += y.size(0)

        print(f"Epoch {epoch+1:02d} | Loss {total_loss/len(train_loader):.4f} | Val Acc {correct/total:.4f}")

    torch.save(model.state_dict(), "transformer_event_model3.pt")
    print("Saved model to transformer_event_model3.pt")


# ---------------- Main ----------------

if __name__ == "__main__":
    train_model("permutations_with_workers_3.npy", device="cpu")
