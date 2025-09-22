import os, time, argparse, csv, pathlib
import cv2
import numpy as np
from ultralytics import YOLO
from fusion_stabilizer import FallStabilizer
from notifier import Notifier
import traceback

COOLDOWN_S = 8
FPS_FALLBACK = 30

# --- 트래커 임포트 (ByteTrack 우선, 실패 시 StrongSORT로 대체) ---
TRACKER_NAME = "ByteTrack"
try:
    from boxmot import ByteTrack as _Tracker
except Exception:
    from boxmot import StrongSORT as _Tracker
    TRACKER_NAME = "StrongSORT"


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)  # 교차 영역 하단 경계는 min
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter + 1e-9
    return inter / union


def fallen_conf_from_dets(track_xyxy, dets_np, fallen_cls_idx):
    """같은 프레임의 fallen_person(cls=idx)들 중 IoU가 가장 큰 conf를 가져옴"""
    if dets_np is None or len(dets_np) == 0:
        return 0.0
    mask = dets_np[:, 5].astype(int) == int(fallen_cls_idx)
    cand = dets_np[mask]
    if cand.shape[0] == 0:
        return 0.0
    ious = np.array([iou_xyxy(track_xyxy, c[:4]) for c in cand], dtype=float)
    j = int(np.argmax(ious))
    if ious[j] < 0.1:
        return 0.0
    return float(cand[j, 4])


def create_tracker(fps):
    """ByteTrack/StrongSORT 시그니처 차이를 흡수"""
    try:
        return _Tracker(track_thresh=0.5, match_thresh=0.7, track_buffer=80, frame_rate=fps)
    except TypeError:
        try:
            return _Tracker(det_thresh=0.5, match_thresh=0.7, buffer_size=80, frame_rate=fps)
        except TypeError:
            return _Tracker()


def parse_args():
    ap = argparse.ArgumentParser()
    # 입력/모델
    ap.add_argument("--model", default="/home/sung/runs/detect/train25/weights/best.pt")
    ap.add_argument("--source", default="0", help='0 또는 "video.mp4"')
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--device", default="cpu", help='"cpu" 또는 "cuda:0"')
    ap.add_argument("--fallen_name", default="Fall-Detected", help="모델 names에서 쓰러진 사람 라벨명")
    # (선택) 라벨 인덱스로 직접 지정하고 싶을 때
    ap.add_argument("--fallen_idx", type=int, default=None)

    # 박스 후처리(배경 오탐 억제)
    ap.add_argument("--min_area", type=float, default=0.005, help="프레임 대비 최소 면적 비율")
    ap.add_argument("--min_ar",   type=float, default=0.0,   help="가로/세로 최소비 (0이면 비활성)")
    ap.add_argument("--max_ar",   type=float, default=999.0, help="가로/세로 최대비")
    ap.add_argument("--draw_min", type=float, default=0.00,  help="이 값 미만이면 텍스트만 표시(디버그)")

    # 안정화/게이트
    ap.add_argument("--tau_on",  type=float, default=0.85)
    ap.add_argument("--tau_off", type=float, default=0.40)
    ap.add_argument("--min_frames", type=int, default=12)
    ap.add_argument("--still_sec",  type=float, default=1.5)
    ap.add_argument("--ema",        type=float, default=0.8)
    ap.add_argument("--gate_state", type=float, default=0.55, help="S 임계")
    ap.add_argument("--gate_event", type=float, default=0.60, help="E 임계 (ST-GCN 있을 때)")

    # ST-GCN
    ap.add_argument("--use_stgcn", type=int, default=1)
    ap.add_argument("--stgcn_ckpt", default="events/ckpts/stgcn_fall.pt")
    ap.add_argument("--stgcn_device", default="cpu", help='"cpu" 또는 "cuda:0"')
    ap.add_argument("--stgcn_seq_len", type=int, default=64)
    ap.add_argument("--stgcn_min_buf", type=int, default=12)
    ap.add_argument("--stgcn_imgsz",   type=int, default=960)
    ap.add_argument("--stgcn_pose_conf", type=float, default=0.10)
    ap.add_argument("--show_pose", type=int, default=0, help="1이면 포즈 스켈레톤 표시(디버그)")
    ap.add_argument("--stgcn_box_expand", type=float, default=1.8)

    return ap.parse_args()


def expand_box(xyxy, scale, W, H):
    x1, y1, x2, y2 = map(float, xyxy)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1)
    h = (y2 - y1)
    s = max(w, h) * scale
    nx1 = max(0.0, cx - s / 2)
    ny1 = max(0.0, cy - s / 2)
    nx2 = min(float(W - 1), cx + s / 2)
    ny2 = min(float(H - 1), cy + s / 2)
    return (nx1, ny1, nx2, ny2)


def main():
    args = parse_args()

    # SOURCE 파싱
    SOURCE = args.source
    if isinstance(SOURCE, str) and SOURCE.isdigit():
        SOURCE = int(SOURCE)

    os.makedirs("logs", exist_ok=True)
    os.makedirs("alerts", exist_ok=True)
    method = "with_stgcn" if args.use_stgcn else "no_stgcn"
    # logf = open(f"logs/{method}.csv", "w", buffering=1)
    # print("video_id,ts,tid,p_state,p_event,R,trigger", file=logf)
    path = f"logs/{method}.csv"
    first = not os.path.exists(path)
    logf = open(path, "a", buffering=1)
    if first:
        print("video_id,ts,tid,p_state,p_event,R,trigger", file=logf)

    # 1) YOLO 로드
    model = YOLO(args.model)

    # fallen 클래스 인덱스 확인/결정
    names = getattr(model, "names", None) or getattr(model.model, "names", None)
    if args.fallen_idx is not None:
        fallen_idx = int(args.fallen_idx)
    else:
        if isinstance(names, dict):
            inv = {v: k for k, v in names.items()}
        else:
            inv = {v: i for i, v in enumerate(names)}
        key = args.fallen_name
        if key not in inv:
            inv_l = {str(k).lower(): v for k, v in inv.items()}
            if key.lower() in inv_l:
                fallen_idx = inv_l[key.lower()]
            else:
                raise RuntimeError(f"[ERR] '{args.fallen_name}' not in model names: {names}")
        else:
            fallen_idx = inv[key]
    print(f"[INFO] fallen_name='{args.fallen_name}' -> idx={fallen_idx}")

    # 2) 소스/FPS (FFMPEG 폴백)
    cap = cv2.VideoCapture(SOURCE, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(SOURCE, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {SOURCE}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else FPS_FALLBACK

    # 3) 트래커
    tracker = create_tracker(fps)

    # 4) 안정화 모듈
    stab = FallStabilizer(
        tau_on=args.tau_on, tau_off=args.tau_off, min_frames=args.min_frames,
        still_sec=args.still_sec, ema=args.ema, fps=fps
    )
    notifier = Notifier(webhook_url=None, cooldown_s=COOLDOWN_S)

    # 5) ST-GCN
    stgcn = None
    if args.use_stgcn:
        try:
            from events.stgcn_infer import STGCNInfer
            stgcn = STGCNInfer(
                ckpt=args.stgcn_ckpt, device=args.stgcn_device,
                seq_len=args.stgcn_seq_len, min_buf=args.stgcn_min_buf,
                imgsz=args.stgcn_imgsz, pose_conf=args.stgcn_pose_conf,
                pose_model_path="events/yolov8n-pose.pt",   # ← 요거 추가
                verbose=True
            )

        except Exception as e:
            print(f"[WARN] ST-GCN unavailable ({e}); running without.")
            stgcn = None

    video_id = str(SOURCE)
    print(f"[INFO] Tracker: {TRACKER_NAME}, ST-GCN: {bool(stgcn)}, Model: {args.model}")

    # === CSV logging init (프레임 로그) ===
    pred_csv = f"logs/pred_{'stgcn' if args.use_stgcn else 'no_stgcn'}.csv"
    write_header = not os.path.exists(pred_csv)
    pred_f = open(pred_csv, "a", newline="")
    pred_w = csv.writer(pred_f)
    if write_header:
        pred_w.writerow([
            "image_id", "video_id", "frame", "timestamp", "track_id",
            "x1", "y1", "x2", "y2", "conf", "class", "stgcn_score"
        ])
    t_rel = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts_cap = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if not np.isnan(ts_cap) and ts_cap > 0 and ts_cap < 1e6:
            ts = float(ts_cap)
        else:
            # 프레임 기반 상대시간 (절대시간 쓰지 않음)
            t_rel += (1.0 / max(1.0, fps))
            ts = float(t_rel)

        # --- 1) YOLO 탐지 ---
        res = model(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            _ = tracker.update(np.zeros((0, 6), dtype=float), frame)
            cv2.imshow("demo", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            continue

        dets_np = np.concatenate(
            [boxes.xyxy.cpu().numpy(),
             boxes.conf.cpu().numpy()[:, None],
             boxes.cls.cpu().numpy()[:, None]],
            axis=1
        )  # [x1,y1,x2,y2,conf,cls]

        # --- 2) 클래스/형상 필터 ---
        H, W = frame.shape[:2]
        keep = []
        for x1, y1, x2, y2, cf, cls in dets_np:
            # 클래스: fallen만
            if int(cls) != fallen_idx:
                continue
            # 면적/종횡비 필터
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            area_ratio = (w * h) / (W * H + 1e-9)
            ar = w / h  # 가로/세로
            if area_ratio < args.min_area:
                continue
            if ar < max(0.0001, args.min_ar) or ar > args.max_ar:
                continue
            keep.append([x1, y1, x2, y2, cf, cls])
        dets_np = np.array(keep, dtype=float) if keep else np.zeros((0, 6), float)

        # --- 3) 추적 ---
        # (주의) boxmot의 반환형은 구현에 따라 다릅니다.
        # 여기서는 [x1,y1,x2,y2,id,conf,cls,ind] 형식을 가정합니다.
        tracks = tracker.update(dets_np, frame)

        # --- 4) 트랙별 처리 ---
        for x1, y1, x2, y2, tid, conf, cls, _ in tracks:
            # (a) 트랙 박스 / (b) ST-GCN 크롭 박스
            xyxy_track = (float(x1), float(y1), float(x2), float(y2))
            xyxy_crop  = expand_box(xyxy_track, args.stgcn_box_expand, W, H)

            os.makedirs("debug_crops", exist_ok=True)
            x1c, y1c, x2c, y2c = map(int, xyxy_crop)
            x1c = max(0, x1c); y1c = max(0, y1c); x2c = min(W-1, x2c); y2c = min(H-1, y2c)
            crop = frame[y1c:y2c, x1c:x2c].copy()
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if crop.size > 0 and frame_idx % 10 == 0:
                cv2.imwrite(f"debug_crops/f{frame_idx}_id{int(tid)}.jpg", crop)


            # 상태 점수는 '트랙 박스'로
            p_state = fallen_conf_from_dets(xyxy_track, dets_np, fallen_idx)

            # ST-GCN 확률 (RGB + 정규화 좌표 우선 시도)
            p_event = 0.0
            buf_len = 0
            if stgcn is not None:
                try:
                    x1c, y1c, x2c, y2c = xyxy_crop
                    xyxy_norm = (x1c / W, y1c / H, x2c / W, y2c / H)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    p_event = float(stgcn.push_and_prob(int(tid), rgb, xyxy_norm) or 0.0)
                    if p_event == 0.0:
                        # 백업: BGR + 절대좌표도 한번 더
                        p_event = float(stgcn.push_and_prob(int(tid), frame, xyxy_crop) or 0.0)

                    seq = stgcn.buf.get(int(tid))
                    buf_len = len(seq) if seq is not None else 0

                    if buf_len % 6 == 0:
                        print(f"[DBG] tid={int(tid)} buf={buf_len} p_event={p_event:.3f}")
                except Exception:
                    print("[ST-GCN ERROR]\n" + traceback.format_exc())
                    p_event = 0.0
                    buf_len = 0

            # Late-Fusion 안정화 (트랙 박스 기준)
            trigger, R, still = stab.update(int(tid), p_state, p_event, xyxy_track)

            # 보수적 게이트
            if p_state < args.gate_state:
                trigger = False
            if stgcn is not None and p_event < args.gate_event:
                trigger = False

            # 시각화 (트랙 박스)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            txt = f"ID {int(tid)} R={R:.2f} S={p_state:.2f} E={p_event:.2f} buf={buf_len}"
            color = (0, 255, 0) if max(p_state, p_event) >= args.draw_min else (0, 128, 0)
            cv2.putText(frame, txt, (int(x1), max(15, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if trigger:
                cv2.putText(frame, "FALL ALERT", (int(x1), int(y1) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                notifier.send(frame, int(tid), p_state, p_event, R, xyxy_track, ts,
                              wh=(frame.shape[1], frame.shape[0]), note=f"still={still}")

            # --- 프레임 로그 기록 (CSV) ---
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # 현재 프레임 index
            image_id = f"{video_id}:{frame_idx}"
            stgcn_score = float(p_event) if stgcn is not None else 0.0
            pred_w.writerow([image_id, video_id, frame_idx, ts, int(tid),
                             float(x1), float(y1), float(x2), float(y2),
                             float(conf), int(cls), stgcn_score])

            # 기존 텍스트 로그(이벤트 상태 요약)
            print(f"{video_id},{ts:.3f},{int(tid)},{p_state:.3f},{p_event:.3f},{R:.3f},{int(trigger)}", file=logf)

        # 키 처리
        cv2.imshow("demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            with open("logs/markers.csv", "a") as f:
                f.write(f"{video_id},{ts:.3f},start\n")
        elif key == ord('e'):
            with open("logs/markers.csv", "a") as f:
                f.write(f"{video_id},{ts:.3f},end\n")
        elif key == ord('n'):
            with open("logs/markers.csv", "a") as f:
                f.write(f"{video_id},{ts:.3f},nonfall\n")

    logf.close()
    pred_f.close()   # CSV 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


