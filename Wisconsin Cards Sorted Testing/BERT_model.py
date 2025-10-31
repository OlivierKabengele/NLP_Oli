import math
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.b = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        return self.g * (x - mean) / torch.sqrt(var + self.eps) + self.b

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        B, T, D = x.shape
        q = self.to_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = self.to_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = self.to_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            # attn_mask: (B, T) boolean where True=token exists
            # convert to (B,1,1,T) to broadcast
            mask = attn_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(B, T, D)
        return self.to_out(out), attn

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads, dropout)
        self.ln2 = LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, attn_mask=None):
        a, attn_map = self.attn(self.ln1(x), attn_mask)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x, attn_map

class BERTEncoder(nn.Module):   # Uses an extra MASK token id appended to vocab.
    def __init__(self, vocab_size=71, dim=128, n_layers=4, n_heads=8, mlp_dim=512, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads, mlp_dim, dropout) for _ in range(n_layers)])
        self.ln_f = LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x, attention_mask=None, return_attn=False, return_hidden=False):
        # x: (B, T)
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(x) + self.pos_emb(positions)
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attention_mask)
            attn_maps.append(attn)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, V)
        if return_attn and return_hidden:
            return logits, attn_maps, x
        if return_attn:
            return logits, attn_maps
        if return_hidden:
            return logits, x
        return logits

class FeatureComparator(nn.Module): # checking per-feature similarities.
    def __init__(self, d_model: int, d_feat: int = 64):
        super().__init__()
        self.d_feat = d_feat
        self.card_proj = nn.ModuleList([nn.Linear(d_model, d_feat) for _ in range(3)])
        self.query_proj = nn.ModuleList([nn.Linear(d_model, d_feat) for _ in range(3)])
        # gate input will be [q_mean (D) , cards_mean (D) , sig_stat (d_feat)] -> total 2*D + d_feat
        gate_input_dim = 2 * d_model + d_feat
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, 128),  # receives [q_mean, cards_mean, sig_stat]
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, cards_sub_emb: torch.Tensor, q_sub_emb: torch.Tensor):
        # cards_sub_emb: (B, 4, 3, D) where the 3 subtokens correspond to (color,shape,number)
        # q_sub_emb: (B, 3, D)
        B = cards_sub_emb.size(0)
        sims = []
        for f in range(3):            # take the f-th subtoken for each card: (B,4,D)
            c_f = cards_sub_emb[:, :, f, :]
            q_f = q_sub_emb[:, f, :]
            c_proj = self.card_proj[f](c_f)  # (B,4,d_f)
            q_proj = self.query_proj[f](q_f).unsqueeze(1)  # (B,1,d_f)
            sim_f = (q_proj * c_proj).sum(-1) / math.sqrt(self.d_feat)  # (B,4)
            sims.append(sim_f)
        sims = torch.stack(sims, dim=1)  # (B,3,4)  # build gate input from query mean and card summary
        cards_mean = cards_sub_emb.mean(dim=(1, 2))  # (B, D)
        q_mean = q_sub_emb.mean(dim=1)  # (B, D)  # mean abs over cards of the sum of subtoken projections
        sig = (cards_sub_emb.sum(dim=2) @ self.query_proj[0].weight.t()) / cards_sub_emb.size(1)
        sig_stat = sig.abs().mean(dim=1, keepdim=False) if sig.dim() == 3 else sig.abs().mean(dim=1)
        gate_in = torch.cat([q_mean, cards_mean, sig_stat], dim=-1)
        gate_logits = self.gate(gate_in)
        gate = torch.softmax(gate_logits, dim=-1)
        final = (gate.unsqueeze(-1) * sims).sum(dim=1)  # (B,4)
        return final, sims, gate_logits

if __name__ == '__main__':
    m = BERTEncoder()
    x = torch.randint(0,71,(2,8))
    out = m(x, attention_mask=torch.ones(2,8,dtype=torch.bool))
    print('logits', out.shape)