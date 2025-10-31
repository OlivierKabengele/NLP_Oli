import numpy as np
import torch
from torch.utils.data import Dataset
try:
    from bert_Token_gen import WCST
except Exception:
    WCST = None

MASK_ID = 70 # We add one MASK token to vocab: default mask_id = 70 (vocab_size must be 71)
COLOR_BASE = 71 # Subtoken bases: we append color/shape/number subtokens after existing vocab (0..70)
SHAPE_BASE = 75 # colors: 71..74, shapes:75..78, numbers:79..82 -> vocab_size = 83
NUMBER_BASE = 79

class WCSTBertDataset(Dataset):#Wrap WCST generator and produce masked LM example(input_tokens_masked,target_tokens, 
    def __init__(self, n_samples=1000, batch_fetch_size=16, seed=0, mask_prob=0.15):  # AND attention_mask)
        self.n = n_samples
        self.batch_fetch_size = batch_fetch_size
        self.pool = []
        self.features = []
        self.mask_prob = mask_prob
        self.rng = np.random.RandomState(seed)
        if WCST is None:
            self._gen_fallback()
        else:
            self._gen_from_wcst()

    def _gen_from_wcst(self): # make the WCST generator switch contexts with some probability per batch
        self.w = WCST(self.batch_fetch_size, switch_prob=0.2) # TO avoid long runs with the same active feature
        while len(self.pool) < self.n:
            ctx_batch, q_batch = next(self.w.gen_batch())# ctx_batch shape (B, 8): 
            for i in range(ctx_batch.shape[0]):          # [c1,c2,c3,c4,example,68,label(64..67),69]
                seq = ctx_batch[i].astype(int).tolist()
                self.pool.append(seq) # store the active feature used to generate this batch/trial
                self.features.append(int(self.w.category_feature)) # WCST stores the active feature in self.w.category_feature
                if len(self.pool) >= self.n:
                    break

    def _gen_fallback(self):
        rng = self.rng
        while len(self.pool) < self.n:
            cats = rng.choice(64, 4, replace=False).tolist()
            example = int(rng.choice([c for c in range(64) if c not in cats]))
            label_idx = rng.randint(0,4)
            seq = cats + [example, 68, 64 + label_idx, 69]
            self.pool.append(seq)            # fallback: unknown feature, set to 0
            self.features.append(0)

    def __len__(self):
        return self.n

    def _mask_tokens(self, tokens):     # tokens: list[int]
        T = len(tokens)
        input_tokens = tokens.copy()
        target = [-100] * T
        for i in range(T):
            if self.rng.rand() < self.mask_prob:             # mask this position
                prob = self.rng.rand()
                target[i] = tokens[i]
                if prob < 0.8:
                    input_tokens[i] = MASK_ID
                elif prob < 0.9:                    # random token from vocab (excluding original)
                    r = self.rng.randint(0,83)
                    if r == tokens[i]:
                        r = (r + 1) % 83
                    input_tokens[i] = int(r)
                else:                    # keep original token (10% chance)
                    input_tokens[i] = tokens[i]
        return input_tokens, target

    def __getitem__(self, idx):
        tokens = self.pool[idx]    # expand each card token (<64) into three subtokens: color, shape, number
        expanded = []
        for t in tokens:
            if int(t) < 64:
                card_idx = int(t)
                color_idx = card_idx // 16
                shape_idx = (card_idx % 16) // 4
                number_idx = card_idx % 4
                expanded.extend([COLOR_BASE + color_idx, SHAPE_BASE + shape_idx, NUMBER_BASE + number_idx])
            else:
                expanded.append(int(t))

        input_tokens, target = self._mask_tokens(expanded)
        attention_mask = [1] * len(input_tokens) 
        try:
            label_token = int(tokens[6])
            label_idx = int(label_token - 64)
        except Exception:
            label_idx = 0
        return (
            torch.tensor(input_tokens, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.bool),
            torch.tensor(self.features[idx], dtype=torch.long),
            torch.tensor(label_idx, dtype=torch.long),
        )

    @staticmethod
    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        masks = torch.stack([b[2] for b in batch])
        feats = torch.stack([b[3] for b in batch])
        labels = torch.stack([b[4] for b in batch])
        return xs, ys, masks, feats, labels


if __name__ == '__main__':
    ds = WCSTBertDataset(32, batch_fetch_size=8, seed=0)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=8, collate_fn=WCSTBertDataset.collate)
    for x, y, m, f, lbl in dl:
        print(x.shape, y.shape, m.shape, f.shape, lbl.shape)
        print('sample in', x[0].tolist())
        print('sample tgt', y[0].tolist())
        print('sample feat', f[0].item(), 'sample label', lbl[0].item())
        break
