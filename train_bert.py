import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import sklearn  # general import
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bert_model import BERTEncoder, FeatureComparator
from bert_dataset import WCSTBertDataset, COLOR_BASE, SHAPE_BASE, NUMBER_BASE

try:
    from wcst_simulator import WCST
except Exception:
    WCST = None

MASK_ID = 70


def masked_accuracy(logits, targets, ignore_index=-100):
    preds = logits.argmax(dim=-1)
    mask = targets != ignore_index
    if mask.sum() == 0:
        return 0.0
    correct = (preds == targets) & mask
    return float(correct.sum().item() / mask.sum().item())


def evaluate(loader, encoder, comparator, device, criterion):
    encoder.eval()
    comparator.eval()
    total_loss = 0.0
    total_lm_acc = 0.0
    total_feat_acc = 0
    total_cat_acc = 0
    total_samples = 0
    n_batches = 0

    conf_cat = torch.zeros((4, 4), dtype=torch.long)
    conf_feat = torch.zeros((3, 3), dtype=torch.long)
    mispredictions = []
    max_misp = 200

    colours = ['red', 'blue', 'green', 'yellow']
    shapes = ['circle', 'square', 'star', 'cross']
    quantities = ['1', '2', '3', '4']

    with torch.no_grad():
        for x, y, m, feat, lbl in loader:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            feat = feat.to(device)
            lbl = lbl.to(device)

            logits, hidden = encoder(x, attention_mask=m, return_hidden=True)
            B, T, V = logits.shape
            lm_loss = criterion(logits.view(B * T, V), y.view(B * T))
            lm_acc = masked_accuracy(logits, y)

            D = hidden.size(-1)
            cards_sub = hidden[:, 0: 4 * 3, :].reshape(B, 4, 3, D)
            q_sub = hidden[:, 4 * 3: 4 * 3 + 3, :]
            final_logits, sims, gate_logits = comparator(cards_sub, q_sub)
            feat_pred = gate_logits.argmax(dim=1)
            cat_pred = final_logits.argmax(dim=1)

            total_correct_cat = (cat_pred == lbl).sum().item()
            total_correct_feat = (feat_pred == feat).sum().item()
            total_feat_acc += total_correct_feat
            total_cat_acc += total_correct_cat
            total_samples += B

            lbl_np = lbl.detach().cpu().numpy()
            cat_np = cat_pred.detach().cpu().numpy()
            feat_np = feat.detach().cpu().numpy()
            featp_np = feat_pred.detach().cpu().numpy()

            for t, p in zip(lbl_np.tolist(), cat_np.tolist()):
                conf_cat[int(t), int(p)] += 1
            for t, p in zip(feat_np.tolist(), featp_np.tolist()):
                conf_feat[int(t), int(p)] += 1

            mismask = (cat_pred != lbl).cpu().numpy()
            x_np = x.detach().cpu().numpy()
            for i in range(B):
                if mismask[i] and len(mispredictions) < max_misp:
                    toks = x_np[i].tolist()
                    cards_readable = []
                    try:
                        for j in range(4):
                            a = toks[3 * j]
                            b = toks[3 * j + 1]
                            c = toks[3 * j + 2]
                            color_idx = int(a - COLOR_BASE)
                            shape_idx = int(b - SHAPE_BASE)
                            number_idx = int(c - NUMBER_BASE)
                            cards_readable.append([colours[color_idx], shapes[shape_idx], quantities[number_idx]])
                    except Exception:
                        cards_readable = toks[:12]
                    mispredictions.append({
                        'cards': cards_readable,
                        'true_cat': int(lbl_np[i]),
                        'pred_cat': int(cat_np[i]),
                        'true_feat': int(feat_np[i]),
                        'pred_feat': int(featp_np[i])
                    })

            total_loss += lm_loss.item()
            total_lm_acc += lm_acc
            n_batches += 1

    feat_sample_acc = float(total_feat_acc / total_samples) if total_samples > 0 else 0.0
    cat_sample_acc = float(total_cat_acc / total_samples) if total_samples > 0 else 0.0

    return (
        total_loss / max(1, n_batches),
        total_lm_acc / max(1, n_batches),
        feat_sample_acc,
        cat_sample_acc,
        conf_feat,
        conf_cat,
        mispredictions,
    )


def compute_roc_auc(loader, encoder, comparator, device):
    """Compute multiclass ROC AUC (macro, OVR) on a dataloader using sklearn if available.

    Returns float or None if sklearn not available or computation fails.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None

    encoder.eval()
    comparator.eval()
    y_trues = []
    y_scores = []
    with torch.no_grad():
        for x, y, m, feat, lbl in loader:
            x = x.to(device)
            m = m.to(device)
            logits, hidden = encoder(x, attention_mask=m, return_hidden=True)
            D = hidden.size(-1)
            cards_sub = hidden[:, 0:4 * 3, :].reshape(x.size(0), 4, 3, D)
            q_sub = hidden[:, 4 * 3:4 * 3 + 3, :]
            final_logits, sims, gate_logits = comparator(cards_sub, q_sub)
            probs = torch.softmax(final_logits, dim=-1).detach().cpu().numpy()
            y_scores.append(probs)
            y_trues.append(lbl.detach().cpu().numpy())

    if len(y_trues) == 0:
        return None
    y_scores = np.concatenate(y_scores, axis=0)
    y_trues = np.concatenate(y_trues, axis=0)
    # build one-hot ground truth
    try:
        y_true_oh = np.eye(y_scores.shape[1])[y_trues]
        roc = float(roc_auc_score(y_true_oh, y_scores, average='macro', multi_class='ovr'))
        return roc
    except Exception:
        return None


if __name__ == '__main__':
    # small training run to exercise model + plot loss
    full_ds = WCSTBertDataset(2000, batch_fetch_size=16, seed=0)
    n = len(full_ds)
    indices = np.arange(n)
    np.random.shuffle(indices)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = indices[:n_train]
    val_idx = indices[n_train: n_train + n_val]
    test_idx = indices[n_train + n_val:]

    from torch.utils.data import Subset
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=32, collate_fn=WCSTBertDataset.collate, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, collate_fn=WCSTBertDataset.collate)
    test_loader = DataLoader(test_ds, batch_size=64, collate_fn=WCSTBertDataset.collate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = BERTEncoder(vocab_size=83, dim=128, n_layers=4, n_heads=4, mlp_dim=256, max_seq_len=64).to(device)
    comparator = FeatureComparator(d_model=128, d_feat=64).to(device)
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(comparator.parameters()), lr=3e-4, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
    feat_criterion = torch.nn.CrossEntropyLoss()
    cat_criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    epochs = 29
    # history tracks losses and category accuracies per epoch
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_cat_acc': [],
        'val_cat_acc': [],
        # store validation confusion-derived metric (macro-F1)
        'val_conf_macro_f1': []
    }
    best_val_loss = float('inf')
    best_path = '/home/olik/NLP/Project_NLP/best_bert_fixed.pt'

    print('Starting training (epochs=%d, train_batches=%d)' % (epochs, len(train_loader)))
    for epoch in range(1, epochs + 1):
        encoder.train()
        comparator.train()
        total_loss = 0.0
        total_lm_acc = 0.0
        total_feat_acc = 0.0
        total_cat_acc = 0.0
        count = 0
        for x, y, m, feat, lbl in train_loader:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            feat = feat.to(device)
            lbl = lbl.to(device)
            logits, hidden = encoder(x, attention_mask=m, return_hidden=True)
            B, T, V = logits.shape
            lm_loss = criterion(logits.view(B * T, V), y.view(B * T))
            D = hidden.size(-1)
            cards_sub = hidden[:, 0:4 * 3, :].reshape(B, 4, 3, D)
            q_sub = hidden[:, 4 * 3:4 * 3 + 3, :]
            final_logits, sims, gate_logits = comparator(cards_sub, q_sub)
            feat_loss = feat_criterion(gate_logits, feat)
            cat_loss = cat_criterion(final_logits, lbl)
            alpha_feat = 0.5
            alpha_cat = 0.5
            loss = lm_loss + alpha_feat * feat_loss + alpha_cat * cat_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(comparator.parameters()), 1.0)
            optimizer.step()
            total_loss += lm_loss.item()
            total_lm_acc += masked_accuracy(logits, y)
            total_feat_acc += (gate_logits.argmax(dim=1) == feat).float().mean().item()
            total_cat_acc += (final_logits.argmax(dim=1) == lbl).float().mean().item()
            count += 1
        scheduler.step()

        train_loss = total_loss / max(1, count)
        train_lm_acc = total_lm_acc / max(1, count)
        train_feat_acc = total_feat_acc / max(1, count)
        train_cat_acc = total_cat_acc / max(1, count)

        val_loss, val_lm_acc, val_feat_acc, val_cat_acc, val_conf_feat, val_conf_cat, val_misp = evaluate(
            val_loader, encoder, comparator, device, criterion
        )

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        # store category accuracy (not LM accuracy) for training/validation
        history['train_cat_acc'].append(train_cat_acc)
        history['val_cat_acc'].append(val_cat_acc)

        print(
            f'Epoch {epoch}/{epochs} - train_lm_loss={train_loss:.4f} train_lm_acc={train_lm_acc:.4f} | '
            f'val_lm_loss={val_loss:.4f} val_lm_acc={val_lm_acc:.4f}'
        )
        print(
            f'    train_feat_acc={train_feat_acc:.4f} train_cat_acc={train_cat_acc:.4f} | '
            f'val_feat_acc={val_feat_acc:.4f} val_cat_acc={val_cat_acc:.4f}'
        )

        # If the underlying WCST generator is present on the full dataset, decode one trial,
        # run the model on that single example and print true label, predicted label, and classification loss.
        try:
            if hasattr(full_ds, 'w') and full_ds.w is not None:
                try:
                    ctx_batch, q_batch = next(full_ds.w.gen_batch())
                    # take first trial in the batch
                    seq = ctx_batch[0].astype(int).tolist()
                    # decode human-readable cards (same logic as WCSTBertDataset)
                    colours = ['red', 'blue', 'green', 'yellow']
                    shapes = ['circle', 'square', 'star', 'cross']
                    quantities = ['1', '2', '3', '4']
                    decoded = []
                    expanded = []
                    for t in seq:
                        if int(t) < 64:
                            card_idx = int(t)
                            color_idx = card_idx // 16
                            shape_idx = (card_idx % 16) // 4
                            number_idx = card_idx % 4
                            decoded.append([colours[color_idx], shapes[shape_idx], quantities[number_idx]])
                            expanded.extend([COLOR_BASE + color_idx, SHAPE_BASE + shape_idx, NUMBER_BASE + number_idx])
                        else:
                            decoded.append(int(t))
                            expanded.append(int(t))

                    # prepare tensors and run model
                    x_tmp = torch.tensor([expanded], dtype=torch.long).to(device)
                    attn_tmp = torch.ones_like(x_tmp, dtype=torch.bool).to(device)
                    with torch.no_grad():
                        logits_tmp, hidden_tmp = encoder(x_tmp, attention_mask=attn_tmp, return_hidden=True)
                        D = hidden_tmp.size(-1)
                        cards_sub_tmp = hidden_tmp[:, 0:4 * 3, :].reshape(1, 4, 3, D)
                        q_sub_tmp = hidden_tmp[:, 4 * 3:4 * 3 + 3, :]
                        final_logits_tmp, sims_tmp, gate_logits_tmp = comparator(cards_sub_tmp, q_sub_tmp)
                        pred_cat = int(final_logits_tmp.argmax(dim=1).item())
                        # true label token stored at position 6 in original seq (64..67)
                        try:
                            true_label = int(seq[6] - 64)
                        except Exception:
                            true_label = None
                        # classification loss for this sample
                        try:
                            if true_label is not None:
                                tmp_lbl = torch.tensor([true_label], dtype=torch.long).to(device)
                                closs = float(cat_criterion(final_logits_tmp, tmp_lbl).item())
                            else:
                                closs = None
                        except Exception:
                            closs = None

                    tl_str = f"C{true_label+1}" if true_label is not None else "None"
                    pl_str = f"C{pred_cat+1}"
                    print('Decoded trial (first sample):', decoded)
                    print('Feature for Classification: ', full_ds.w.category_feature)
                    print(f'True label: {tl_str} Predicted label: {pl_str} Class loss: {closs}')
                except Exception as e:
                    print('Could not fetch/run trial from WCST generator:', e)
        except Exception:
            pass

        np.save(f'/home/olik/NLP/Project_NLP/val_conf_cat_epoch{epoch}.npy', val_conf_cat.numpy())
        np.save(f'/home/olik/NLP/Project_NLP/val_conf_feat_epoch{epoch}.npy', val_conf_feat.numpy())
        try:
            with open(f'/home/olik/NLP/Project_NLP/val_mispred_epoch{epoch}.json', 'w') as f:
                json.dump(val_misp, f, indent=2)
        except Exception:
            pass

        # compute confusion-derived metrics (precision/recall/F1) for category confusion matrix
        try:
            conf = val_conf_cat.numpy()
            tp = conf.diagonal().astype(float)
            pred_sum = conf.sum(axis=0).astype(float)  # predicted per class
            true_sum = conf.sum(axis=1).astype(float)  # true per class
            with np.errstate(divide='ignore', invalid='ignore'):
                precision = np.where(pred_sum > 0, tp / pred_sum, 0.0)
                recall = np.where(true_sum > 0, tp / true_sum, 0.0)
                f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
            macro_f1 = float(np.nanmean(f1))
        except Exception:
            precision = recall = f1 = np.zeros(4)
            macro_f1 = 0.0

        history['val_conf_macro_f1'].append(macro_f1)
        print(f'Val confusion macro-F1: {macro_f1:.4f} | per-class F1: {f1}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'encoder': encoder.state_dict(), 'comparator': comparator.state_dict()}, best_path)

    # final test
    try:
        sd = torch.load(best_path)
        encoder.load_state_dict(sd.get('encoder', sd))
        comparator.load_state_dict(sd.get('comparator', {}))
        print('Loaded best model from', best_path)
    except Exception:
        print('No best model saved, using current model')

    test_loss, test_lm_acc, test_feat_acc, test_cat_acc, test_conf_feat, test_conf_cat, test_misp = evaluate(
        test_loader, encoder, comparator, device, criterion
    )
    print(
        f'Test - loss={test_loss:.4f} lm_acc={test_lm_acc:.4f} feat_acc={test_feat_acc:.4f} cat_acc={test_cat_acc:.4f}'
    )
    print(f'Final test category accuracy: {test_cat_acc:.4f}')
    np.save('/home/olik/NLP/Project_NLP/test_conf_cat.npy', test_conf_cat.numpy())
    np.save('/home/olik/NLP/Project_NLP/test_conf_feat.npy', test_conf_feat.numpy())
    try:
        with open('/home/olik/NLP/Project_NLP/test_mispred.json', 'w') as f:
            json.dump(test_misp, f, indent=2)
    except Exception:
        pass

    # plot and save
    epochs_range = list(range(1, len(history['train_loss']) + 1))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='train_loss')
    plt.plot(epochs_range, history['val_loss'], label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    # plot category accuracy
    plt.plot(epochs_range, history['train_cat_acc'], label='train_cat_acc')
    plt.plot(epochs_range, history['val_cat_acc'], label='val_cat_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.title('Accuracy')
    plt.tight_layout()
    out_path = '/home/olik/NLP/Project_NLP/train_loss_fixed.png'
    plt.savefig(out_path)
    print(f"Saved training plots to: {out_path}")

# save metrics summary (losses and category accuracies) and final test accuracy
metrics = {
    'history': history,
    'final_test': {
        'test_loss': float(test_loss),
        'test_lm_acc': float(test_lm_acc),
        'test_feat_acc': float(test_feat_acc),
        'test_cat_acc': float(test_cat_acc)
    }
}
try:
    with open('/home/olik/NLP/Project_NLP/metrics.json', 'w') as f:
        # also compute test confusion macro-F1 and include in metrics
        try:
            conft = test_conf_cat.numpy()
            tp_t = conft.diagonal().astype(float)
            pred_sum_t = conft.sum(axis=0).astype(float)
            true_sum_t = conft.sum(axis=1).astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                prec_t = np.where(pred_sum_t > 0, tp_t / pred_sum_t, 0.0)
                rec_t = np.where(true_sum_t > 0, tp_t / true_sum_t, 0.0)
                f1_t = np.where((prec_t + rec_t) > 0, 2 * prec_t * rec_t / (prec_t + rec_t), 0.0)
            metrics['final_test']['test_conf_macro_f1'] = float(np.nanmean(f1_t))
            metrics['final_test']['test_conf_per_class_f1'] = f1_t.tolist()
        except Exception:
            metrics['final_test']['test_conf_macro_f1'] = None
            metrics['final_test']['test_conf_per_class_f1'] = None
        json.dump(metrics, f, indent=2)
    print('Saved metrics summary to /home/olik/NLP/Project_NLP/metrics.json')
except Exception:
    pass

    # print the last 3 validation confusion matrices (category) and their macro-F1s
    try:
        epoch_count = len(history['val_loss'])
        last_idxs = list(range(max(1, epoch_count - 2), epoch_count + 1))
        print('\nLast validation category confusion matrices (last 3 epochs):')
        for e in last_idxs:
            try:
                arr = np.load(f'/home/olik/NLP/Project_NLP/val_conf_cat_epoch{e}.npy')
                print(f' Epoch {e}:')
                print(arr)
                # attempt to show the macro-f1 stored in history if available
                try:
                    mf1 = history.get('val_conf_macro_f1', [None])
                    print('  saved val macro-F1:', mf1[e-1] if e-1 < len(mf1) else None)
                except Exception:
                    pass
            except Exception as _:
                print(f' Epoch {e}: (file missing)')
    except Exception:
        pass

    # compute ROC-AUC on test set (if sklearn available)
    try:
        roc = compute_roc_auc(test_loader, encoder, comparator, device)
        if roc is not None:
            print(f'Final test ROC-AUC (macro OVR): {roc:.4f}')
            try:
                metrics['final_test']['test_roc_auc'] = roc
                with open('/home/olik/NLP/Project_NLP/metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)
            except Exception:
                pass
        else:
            print('ROC-AUC: sklearn not available or computation failed (skipped)')
    except Exception:
        print('ROC-AUC computation failed')
