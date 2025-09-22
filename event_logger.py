#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
event_logger.py
프레임 단위 탐지/추적 로그를 트랙 기반 *이벤트* CSV로 변환.

입력 CSV(프레임 레벨) 스키마:
  image_id,video_id,frame,timestamp,track_id,x1,y1,x2,y2,conf,class,stgcn_score

출력 CSV(이벤트 레벨) 스키마:
  video_id,track_id,t_start,t_end,score,label

baseline(미적용)은 conf로, ST-GCN은 stgcn_score로 이벤트를 만들고,
구간 점수는 기본적으로 max를 사용합니다.

사용 예:
  python event_logger.py --pred_csv logs/pred_no_stgcn.csv --out_csv logs/events_no_stgcn.csv --method baseline
  python event_logger.py --pred_csv logs/pred_stgcn.csv  --out_csv logs/events_stgcn.csv  --method stgcn
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Literal, Tuple

def _nonmax_time_merge(segments: np.ndarray, scores: np.ndarray, tiou_thresh: float=0.5):
    """시간 구간에 대한 간단한 NMM(merge). 겹치면 구간 확장, 점수는 max."""
    if len(segments) == 0:
        return segments, scores
    order = np.argsort(-scores)
    segs, scrs = segments[order], scores[order]
    kept, kept_scores = [], []
    for s, sc in zip(segs, scrs):
        merged = False
        for i in range(len(kept)):
            inter = max(0.0, min(s[1], kept[i][1]) - max(s[0], kept[i][0]))
            uni = (s[1]-s[0]) + (kept[i][1]-kept[i][0]) - inter + 1e-9
            tiou = inter / uni
            if tiou >= tiou_thresh:
                kept[i] = np.array([min(kept[i][0], s[0]), max(kept[i][1], s[1])])
                kept_scores[i] = max(kept_scores[i], sc)
                merged = True
                break
        if not merged:
            kept.append(s.copy())
            kept_scores.append(sc)
    return np.array(kept), np.array(kept_scores)

def trackwise_eventize(df: pd.DataFrame,
                       method: Literal["baseline","stgcn"]="baseline",
                       min_len: float = 0.3,
                       gap: float = 0.4,
                       agg: Literal["max","mean"]="max",
                       conf_thresh: float = 0.25,
                       stgcn_thresh: float = 0.50,
                       tiou_merge: float = 0.5) -> pd.DataFrame:
    """
    프레임 → 트랙 이벤트 변환.
    min_len: 최소 이벤트 길이(s)
    gap:     내부 결합 허용 간격(s)
    """
    req = {"video_id","frame","timestamp","track_id","x1","y1","x2","y2","conf"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    if method == "baseline":
        pos_mask = df["conf"] >= conf_thresh
        frame_score = df["conf"].values
    else:
        if "stgcn_score" not in df.columns:
            raise ValueError("ST-GCN method needs 'stgcn_score'")
        pos_mask = df["stgcn_score"] >= stgcn_thresh
        frame_score = df["stgcn_score"].values

    rows = []
    for (vid, tid), g in df[pos_mask].groupby(["video_id","track_id"]):
        g = g.sort_values("timestamp")
        t = g["timestamp"].values
        s = g["stgcn_score"].values if method=="stgcn" else g["conf"].values

        # 연속 프레임을 구간화
        segs = []
        start, cur = t[0], t[0]
        buf = [s[0]]
        for i in range(1, len(t)):
            if t[i] - cur <= gap:
                cur = t[i]; buf.append(s[i])
            else:
                if cur - start >= min_len:
                    segs.append((start, cur, np.max(buf) if agg=="max" else np.mean(buf)))
                start, cur, buf = t[i], t[i], [s[i]]
        if cur - start >= min_len:
            segs.append((start, cur, np.max(buf) if agg=="max" else np.mean(buf)))

        if not segs:
            continue
        segs = np.array(segs)
        seg_xy, seg_sc = segs[:, :2], segs[:, 2]
        seg_xy, seg_sc = _nonmax_time_merge(seg_xy, seg_sc, tiou_merge)

        for (ts, te), sc in zip(seg_xy, seg_sc):
            rows.append([vid, int(tid), float(ts), float(te), float(sc), 1])

    return pd.DataFrame(rows, columns=["video_id","track_id","t_start","t_end","score","label"])

def save_events(pred_csv: str, out_csv: str, method: Literal["baseline","stgcn"]="baseline", **kwargs):
    df = pd.read_csv(pred_csv)
    ev = trackwise_eventize(df, method=method, **kwargs)
    ev.to_csv(out_csv, index=False)
    print(f"[event_logger] wrote {len(ev)} events -> {out_csv}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--method", choices=["baseline","stgcn"], default="baseline")
    ap.add_argument("--min_len", type=float, default=0.3)
    ap.add_argument("--gap", type=float, default=0.4)
    ap.add_argument("--agg", choices=["max","mean"], default="max")
    ap.add_argument("--conf_thresh", type=float, default=0.25)
    ap.add_argument("--stgcn_thresh", type=float, default=0.50)
    ap.add_argument("--tiou_merge", type=float, default=0.5)
    args = ap.parse_args()
    save_events(args.pred_csv, args.out_csv, args.method,
                min_len=args.min_len, gap=args.gap, agg=args.agg,
                conf_thresh=args.conf_thresh, stgcn_thresh=args.stgcn_thresh,
                tiou_merge=args.tiou_merge)
