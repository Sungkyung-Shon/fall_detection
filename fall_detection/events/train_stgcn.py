# events/train_stgcn.py
import os, glob, argparse, random, numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from .models.stgcn import STGCN

def set_seed(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def subject_from_rel(rel):
    # "Subject 3/Fall/07" -> "Subject 3"
    parts = rel.replace('\\','/').split('/')
    return parts[0] if parts else 'Unknown'

def uniform_sample_idx(T, L):
    if T >= L:
        return np.linspace(0, T-1, L).astype(int)
    else:
        idx = np.arange(T)
        pad = np.full(L-T, T-1, dtype=int)
        return np.concatenate([idx, pad])

def normalize_xy(seq):  # seq: [T,17,3] (x,y,score in pixel coords)
    xy = seq[...,:2].copy()      # [T,17,2]
    sc = seq[..., 2:3].copy()    # [T,17,1]

    # 중심: 좌/우 hip 평균
    center = (xy[:,11,:] + xy[:,12,:]) / 2.0  # [T,2]
    xy = xy - center[:,None,:]

    # 스케일: 어깨폭 + 엉덩이폭 (안전장치)
    shoulder = np.linalg.norm(xy[:,5,:] - xy[:,6,:], axis=1)  # [T]
    hip      = np.linalg.norm(xy[:,11,:] - xy[:,12,:], axis=1)
    scale = (shoulder + hip) / 2.0
    scale[scale < 1.0] = 1.0
    xy = xy / scale[:,None,None]

    # [-1,1] 범위 클리핑(너무 튀는 값 방지)
    xy = np.clip(xy, -3.0, 3.0) / 3.0

    out = np.concatenate([xy, sc], axis=-1)   # [T,17,3]
    return out.astype(np.float32)

class SkeletonSeqNPZ(Dataset):
    def __init__(self, root, split='train', val_subjects=('Subject 3',), seq_len=64):
        self.items = []
        self.seq_len = seq_len
        npzs = sorted(glob.glob(os.path.join(root, '**', 'seq.npz'), recursive=True))
        for p in npzs:
            z = np.load(p)
            rel = str(z['rel'])
            subj = subject_from_rel(rel)
            is_val = subj in val_subjects
            if (split == 'train' and not is_val) or (split == 'val' and is_val):
                self.items.append(p)
        if len(self.items) == 0:
            raise RuntimeError(f"No data for split={split}. Check val_subjects or root path.")

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        z = np.load(self.items[i])
        kpts  = z['kpts'].astype(np.float32)   # [T,17,3]
        label = int(z['label'])
        T = kpts.shape[0]
        idx = uniform_sample_idx(T, self.seq_len)
        seq = kpts[idx]                        # [L,17,3]
        seq = normalize_xy(seq)                # [L,17,3]
        # [C,T,V]
        x = np.transpose(seq, (2,0,1)).astype(np.float32)  # [3,L,17]
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.long)

def train_one_epoch(model, loader, crit, opt, dev):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x,y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        loss_sum += float(loss) * y.size(0)
        pred = logits.argmax(1)
        correct += int((pred == y).sum())
        total += y.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def evaluate(model, loader, crit, dev):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    tp=fp=fn=0
    for x,y in loader:
        x,y = x.to(dev), y.to(dev)
        logits = model(x)
        loss = crit(logits, y)
        loss_sum += float(loss) * y.size(0)
        pred = logits.argmax(1)
        correct += int((pred == y).sum())
        total += y.size(0)
        # F1 for fall=1
        tp += int(((pred==1) & (y==1)).sum())
        fp += int(((pred==1) & (y==0)).sum())
        fn += int(((pred==0) & (y==1)).sum())
    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    return loss_sum/total, correct/total, (prec, rec, f1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sk_root', default='datasets/gmdcsa24_skeleton')
    ap.add_argument('--val_subjects', default='Subject 3', help='쉼표로 여러 개 지정 가능')
    ap.add_argument('--seq_len', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out', default='events/ckpts/stgcn_fall.pt')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    val_subjects = tuple([s.strip() for s in args.val_subjects.split(',') if s.strip()])
    dev = torch.device(args.device if (args.device!='cpu' and torch.cuda.is_available()) else 'cpu')

    train_ds = SkeletonSeqNPZ(args.sk_root, 'train', val_subjects, args.seq_len)
    val_ds   = SkeletonSeqNPZ(args.sk_root, 'val',   val_subjects, args.seq_len)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model = STGCN(in_ch=3, num_class=2, num_nodes=17, t_kernel=9, dropout=0.25).to(dev)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_f1 = 0.0
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, crit, opt, dev)
        va_loss, va_acc, (p,r,f1) = evaluate(model, val_ld, crit, dev)
        print(f"[{ep:03d}] train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} P {p:.3f} R {r:.3f} F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save({'model':model.state_dict(),
                        'meta':{'seq_len':args.seq_len}}, args.out)
            print(f"  -> saved best to {args.out} (F1={best_f1:.3f})")

if __name__ == "__main__":
    main()
