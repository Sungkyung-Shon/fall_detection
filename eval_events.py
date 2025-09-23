import argparse, glob, os, re
import numpy as np
import pandas as pd
import cv2

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sysA', required=True, help='no ST-GCN log csv')
    ap.add_argument('--sysB', required=True, help='with ST-GCN log csv')
    ap.add_argument('--gt_glob', required=True, help='glob for GT CSVs (class,start,end,video_id)')
    ap.add_argument('--videos_dir', default='', help='dir of videos (to sum durations)')
    ap.add_argument('--grace', type=float, default=1.5, help='TP grace seconds after end')
    ap.add_argument('--out', default='logs/metrics.csv', help='where to save table csv')
    return ap.parse_args()

def _try_parse_time(x):
    """support 12.3 or mm:ss or hh:mm:ss"""
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if re.fullmatch(r'\d+(\.\d+)?', s):  # seconds as float
        return float(s)
    parts = [float(p) for p in s.split(':')]
    if len(parts) == 2:
        m, s = parts; return 60*m + s
    if len(parts) == 3:
        h, m, s = parts; return 3600*h + 60*m + s
    return np.nan

def load_sys_log(path):
    # video_id,ts,tid,p_state,p_event,R,trigger
    df = pd.read_csv(path)
    if 'trigger' in df.columns:
        df = df[df['trigger'] == 1]
    if 'video_id' not in df.columns:
        df['video_id'] = 'unknown'
    # 정규화: 파일경로면 파일명(stem)만 쓰기
    df['video_id'] = df['video_id'].astype(str).apply(lambda p: os.path.splitext(os.path.basename(p))[0])
    return df[['video_id','ts']].sort_values(['video_id','ts'])

def load_gt(glob_pat):
    frames = []
    for fp in glob.glob(glob_pat):
        g = pd.read_csv(fp)
        # flexible column names
        col_vid = [c for c in g.columns if c.lower() in ('video','video_id','video-name','file','filename')]
        col_cls = [c for c in g.columns if c.lower() in ('class','label')]
        col_s   = [c for c in g.columns if c.lower() in ('start','start_time','start sec','start(s)')]
        col_e   = [c for c in g.columns if c.lower() in ('end','end_time','end sec','end(s)')]
        if not (col_vid and col_cls and col_s and col_e): 
            continue
        g = g.rename(columns={col_vid[0]:'video_id', col_cls[0]:'class',
                              col_s[0]:'start', col_e[0]:'end'})
        # normalize id to stem
        g['video_id'] = g['video_id'].astype(str).apply(lambda p: os.path.splitext(os.path.basename(p))[0])
        g['start'] = g['start'].apply(_try_parse_time)
        g['end']   = g['end'].apply(_try_parse_time)
        g = g.dropna(subset=['start','end'])
        # fall만
        g = g[g['class'].astype(str).str.lower().str.contains('fall')]
        frames.append(g[['video_id','start','end']])
    if not frames:
        raise RuntimeError('No GT rows parsed. Check --gt_glob and column names.')
    return pd.concat(frames, ignore_index=True)

def durations_from_videos(videos_dir, video_ids):
    dur = {}
    for vid in sorted(set(video_ids)):
        # 후보 파일 찾기
        cand = []
        for ext in ('*.mp4','*.avi','*.mov','*.mkv'):
            cand += glob.glob(os.path.join(videos_dir, f'{vid}{ext[1:]}'))  # exact match
            cand += glob.glob(os.path.join(videos_dir, f'*{vid}*{ext[1:]}'))  # loose
        cand = sorted(set(cand))
        if not cand:
            continue
        path = cand[0]
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if nfr > 0:
            dur[vid] = float(nfr / fps)
    return dur

def metrics(sys_alerts, gt_events, vid2dur, grace=1.5):
    TP=FP=FN=0; lat=[]
    vids = sorted(set(gt_events['video_id']))
    for vid in vids:
        g = gt_events[gt_events['video_id']==vid].sort_values('start').to_numpy()
        s = sys_alerts[sys_alerts['video_id']==vid]['ts'].to_numpy()
        used = np.zeros(len(s), dtype=bool)
        for start, end in g:
            idx = np.where((s>=start) & (s<=end+grace))[0]
            if len(idx):
                TP += 1
                used[idx[0]] = True
                lat.append(max(0.0, s[idx[0]]-start))
            else:
                FN += 1
        FP += int((~used).sum())
    prec = TP/(TP+FP+1e-9); rec = TP/(TP+FN+1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)
    total_sec = float(sum(vid2dur.get(v, 0.0) for v in vids))
    # fallback: 로그 span 합
    if total_sec <= 0:
        for vid in vids:
            s = sys_alerts[sys_alerts['video_id']==vid]['ts']
            if len(s): total_sec += float(s.max()-s.min())
    far_h = FP / (total_sec/3600.0 + 1e-9) if total_sec>0 else np.nan
    med = float(np.median(lat)) if lat else np.nan
    p90 = float(np.percentile(lat,90)) if lat else np.nan
    p2s = float(np.mean([l<=2.0 for l in lat])) if lat else np.nan
    return dict(TP=TP, FP=FP, FN=FN, precision=prec, recall=rec, f1=f1,
                FAR_per_hour=far_h, median_latency_s=med, p90_latency_s=p90, pct_detect_le2s=p2s)

def main():
    args = parse_args()
    sysA = load_sys_log(args.sysA)
    sysB = load_sys_log(args.sysB)
    gt   = load_gt(args.gt_glob)
    vid2dur = durations_from_videos(args.videos_dir, gt['video_id'].unique()) if args.videos_dir else {}

    mA = metrics(sysA, gt, vid2dur, grace=args.grace)
    mB = metrics(sysB, gt, vid2dur, grace=args.grace)

    table = pd.DataFrame({
        'Metric': ['Event-F1','FAR/h','Median latency (s)','P90 latency (s)','% detected ≤2s','Precision','Recall','TP','FP','FN'],
        'USE_STGCN=False': [mA['f1'], mA['FAR_per_hour'], mA['median_latency_s'], mA['p90_latency_s'],
                            mA['pct_detect_le2s'], mA['precision'], mA['recall'], mA['TP'], mA['FP'], mA['FN']],
        'USE_STGCN=True':  [mB['f1'], mB['FAR_per_hour'], mB['median_latency_s'], mB['p90_latency_s'],
                            mB['pct_detect_le2s'], mB['precision'], mB['recall'], mB['TP'], mB['FP'], mB['FN']]
    })
    # 표시 포맷
    def fmt(x):
        if isinstance(x,(int,np.integer)): return x
        return None if pd.isna(x) else float(x)
    table = table.applymap(fmt)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    table.to_csv(args.out, index=False)
    # 콘솔 출력
    print('\n=== Comparison ===')
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()