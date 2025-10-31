import torch
import numpy as np
from BERT_model import BERTEncoder, FeatureComparator
from bert_SubToken_gen import WCSTBertDataset

# KINDLY REPLACE THE PATH address in ""out_prefix='/home/olik/NLP/Project_NLP/interpret ""

def analyze_interpretability_local(loader, encoder, comparator, device, max_batches=20, out_prefix='/home/olik/NLP/Project_NLP/interpret'):
    encoder.eval()
    comparator.eval()
    acc = {f: None for f in range(3)}
    total_examples = 0
    try:
        for b_ix, (x, y, m, feat, lbl) in enumerate(loader):
            if b_ix >= max_batches:
                break
            x = x.to(device)
            m = m.to(device)
            with torch.no_grad():
                out = encoder(x, attention_mask=m, return_attn=True, return_hidden=True)
                if isinstance(out, tuple) and len(out) == 3:
                    logits, attn_maps, hidden = out
                else:
                    continue

                B = x.size(0)
                n_layers = len(attn_maps)
                n_heads = attn_maps[0].size(1)
                for f in range(3):
                    if acc[f] is None:
                        acc[f] = [np.zeros((n_heads,), dtype=float) for _ in range(n_layers)]

                lbl_np = lbl.detach().cpu().numpy()
                for i in range(B):
                    lab = int(lbl_np[i])
                    for f in range(3):
                        pos_q = 4 * 3 + f
                        pos_c = 3 * lab + f
                        for L in range(n_layers):
                            a = attn_maps[L][i].detach().cpu().numpy()
                            w = a[:, pos_q, pos_c]
                            acc[f][L] += w
                total_examples += B

        if total_examples == 0:
            print('No examples processed for interpretability.')
            return None

        head_scores = {}
        for f in range(3):
            arr = np.stack([acc[f][L] / float(total_examples) for L in range(len(acc[f]))], axis=0)
            head_scores[f] = arr
            np.save(f'{out_prefix}_head_scores_feat{f}.npy', arr)

        print('\nInterpretability: top attention heads attending to correct card feature subtokens')
        for f in range(3):
            arr = head_scores[f]
            n_layers, n_heads = arr.shape
            flat = arr.flatten()
            idxs = np.argsort(-flat)
            print(f'Feature {f} top heads:')
            for r in range(min(8, len(flat))):
                idx = idxs[r]
                L = idx // n_heads
                H = idx % n_heads
                print(f'  Rank {r+1}: Layer {L} Head {H} score={arr[L,H]:.4f}')

        try:
            np.savez(f'{out_prefix}_summary.npz', **{f'feat{f}': head_scores[f] for f in head_scores})
        except Exception:
            pass
        return head_scores
    except Exception as e:
        print('analyze_interpretability_local failed:', e)
        return None

def main():
    full_ds = WCSTBertDataset(2000, batch_fetch_size=16, seed=0)
    n = len(full_ds)
    indices = np.arange(n)
    np.random.shuffle(indices)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train: n_train + n_val]
    test_idx = indices[n_train + n_val:]

    from torch.utils.data import Subset, DataLoader
    test_ds = Subset(full_ds, test_idx)
    test_loader = DataLoader(test_ds, batch_size=64, collate_fn=WCSTBertDataset.collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = BERTEncoder(vocab_size=83, dim=128, n_layers=4, n_heads=4, mlp_dim=256, max_seq_len=64).to(device)
    comparator = FeatureComparator(d_model=128, d_feat=64).to(device)

    # load best checkpoint if present
    import os
    ck = '/home/olik/NLP/Project_NLP/best_bert.pt'   #  KINDLY REPLACE THIS PATH WITH YOURS

    if os.path.exists(ck):
        sd = torch.load(ck, map_location=device)
        encoder.load_state_dict(sd.get('encoder', sd))
        comparator.load_state_dict(sd.get('comparator', {}))
        print('Loaded checkpoint', ck)
    else:
        print('No checkpoint at', ck)

    head_scores = analyze_interpretability_local(test_loader, encoder, comparator, device, max_batches=20,
                                                 out_prefix='/home/olik/NLP/Project_NLP/interpret') # KINDLY ADJUST THIS PATH
    if head_scores is None:
        print('Interpretability returned None')
    else:
        for f, arr in head_scores.items():
            print(f'Feature {f} head scores shape: {arr.shape}')

if __name__ == '__main__':
    main()
