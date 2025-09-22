#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_plots.py
Baseline(미적용) vs +ST-GCN 이벤트 비교 시각화.

입력:
- GT 이벤트 CSV: video_id,t_start,t_end,label
- Pred 이벤트 CSV A/B: video_id,track_id,t_start,t_end,score,label

산출:
- PR 곡선(AP 포함) png
- DET 곡선(FPPI vs Recall) png
- FP/FN 막대 png
- (선택) 특정 비디오 타임라인 png
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Tuple
from statsmodels.stats.contingency_tables import mcnemar
from pathlib import Path

def tiou(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1]-a[0]) + (b[1]-b[0]) - inter + 1e-9
    return inter/union

def match_events(gt: pd.DataFrame, pred: pd.DataFrame, tiou_thresh: float=0.5):
    """점수순 그리디 매칭."""
    tp, fp, scores = [], [], []
    gt_taken = defaultdict(set)
    pred_sorted = pred.sort_values("score", ascending=False).reset_index(drop=True)
    for _, r in pred_sorted.iterrows():
        vid = r["video_id"]; p = (r["t_start"], r["t_end"]); sc = r["score"]
        cand = gt[gt["video_id"]==vid].reset_index()
        ok = False
        for _, g in cand.iterrows():
            if g["index"] in gt_taken[vid]:
                continue
            if tiou(p, (g["t_start"], g["t_end"])) >= tiou_thresh:
                gt_taken[vid].add(g["index"])
                tp.append(1); fp.append(0); scores.append(sc); ok=True; break
        if not ok:
            tp.append(0); fp.append(1); scores.append(sc)
    fn = 0
    for vid, g in gt.groupby("video_id"):
        fn += len(g) - len(gt_taken[vid])
    return np.array(tp), np.array(fp), fn, np.array(scores)

def pr_curve_from_events(gt, pred, tiou_thresh=0.5):
    tp, fp, fn, scores = match_events(gt, pred, tiou_thresh)
    order = np.argsort(-scores)
    tp, fp = tp[order], fp[order]
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    recalls = cum_tp / (len(gt) + 1e-9)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-9)
    # 101-pt AP
    ap = 0.0
    for r in np.linspace(0,1,101):
        p = np.max(precisions[recalls>=r]) if np.any(recalls>=r) else 0.0
        ap += p/101.0
    return recalls, precisions, ap, fn

def fppi_recall_curve(gt, pred, minutes_total, tiou_thresh=0.5):
    tp, fp, fn, scores = match_events(gt, pred, tiou_thresh)
    order = np.argsort(-scores)
    tp, fp = tp[order], fp[order]
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    recalls = cum_tp / (len(gt) + 1e-9)
    fppi = cum_fp / (minutes_total + 1e-9)
    return fppi, recalls

def paired_fp_fn(gt, predA, predB, tiou_thresh=0.5):
    _, _, fnA, _ = match_events(gt, predA, tiou_thresh)
    _, _, fnB, _ = match_events(gt, predB, tiou_thresh)
    fpA = len(predA) - (len(gt) - fnA)
    fpB = len(predB) - (len(gt) - fnB)
    return fpA, fnA, fpB, fnB

# -------- Plot helpers --------
def plot_pr(recA, precA, apA, recB, precB, apB, out_png):
    plt.figure()
    plt.plot(recA, precA, label=f"No ST-GCN (AP={apA:.3f})")
    plt.plot(recB, precB, label=f"+ ST-GCN (AP={apB:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Event-level PR (tIoU=0.5)")
    plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")

def plot_det(fppiA, recA, fppiB, recB, out_png):
    plt.figure()
    plt.plot(fppiA, recA, label="No ST-GCN")
    plt.plot(fppiB, recB, label="+ ST-GCN")
    plt.xlabel("FP per Minute (lower is better)"); plt.ylabel("Recall"); plt.title("DET-like Curve")
    plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")

def plot_fp_fn(fpA, fnA, fpB, fnB, out_png):
    plt.figure()
    labels = ["FP","FN"]
    A = [fpA, fnA]; B = [fpB, fnB]
    x = np.arange(2); w = 0.35
    plt.bar(x-w/2, A, width=w, label="No ST-GCN")
    plt.bar(x+w/2, B, width=w, label="+ ST-GCN")
    plt.xticks(x, labels); plt.title("Paired FP/FN (lower is better)")
    plt.legend(); plt.grid(True, axis="y", alpha=0.2)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")

def plot_timeline(gt, predA, predB, video_id, out_png):
    g = gt[gt["video_id"]==video_id]
    a = predA[predA["video_id"]==video_id]
    b = predB[predB["video_id"]==video_id]
    def draw(ax, df, y, label):
        for _, r in df.iterrows():
            ax.plot([r["t_start"], r["t_end"]], [y,y], linewidth=6)
        ax.text(0, y+0.15, label)
    plt.figure()
    ax = plt.gca()
    draw(ax, g, 2.0, "GT")
    draw(ax, a, 1.0, "No ST-GCN")
    draw(ax, b, 0.0, "+ ST-GCN")
    ax.set_xlabel("Time (s)"); ax.set_yticks([]); ax.set_title(f"Event Timeline – {video_id}")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")

# -------- Runner --------
def run_all(gt_events_csv: str, pred_ev_A_csv: str, pred_ev_B_csv: str,
            minutes_total: float, out_dir: str="./compare_out", tiou_thresh: float=0.5):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    gt = pd.read_csv(gt_events_csv)
    A  = pd.read_csv(pred_ev_A_csv)
    B  = pd.read_csv(pred_ev_B_csv)

    recA, precA, apA, _ = pr_curve_from_events(gt, A, tiou_thresh)
    recB, precB, apB, _ = pr_curve_from_events(gt, B, tiou_thresh)
    plot_pr(recA, precA, apA, recB, precB, apB, out / "pr_curve.png")

    fppiA, rA = fppi_recall_curve(gt, A, minutes_total, tiou_thresh)
    fppiB, rB = fppi_recall_curve(gt, B, minutes_total, tiou_thresh)
    plot_det(fppiA, rA, fppiB, rB, out / "det_curve.png")

    fpA, fnA, fpB, fnB = paired_fp_fn(gt, A, B, tiou_thresh)
    plot_fp_fn(fpA, fnA, fpB, fnB, out / "fp_fn.png")
    print(f"[compare] AP: no-STGCN={apA:.3f}, +STGCN={apB:.3f} | FP: {fpA}->{fpB}, FN: {fnA}->{fnB}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=False, help="gt_events.csv")
    ap.add_argument("--a",  required=False, help="events_no_stgcn.csv")
    ap.add_argument("--b",  required=False, help="events_stgcn.csv")
    ap.add_argument("--minutes", type=float, default=1.0)
    ap.add_argument("--out", default="./compare_out")
    ap.add_argument("--tiou", type=float, default=0.5)
    args = ap.parse_args()
    if args.gt and args.a and args.b:
        run_all(args.gt, args.a, args.b, args.minutes, args.out, args.tiou)
