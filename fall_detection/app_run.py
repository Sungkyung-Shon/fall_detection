# YOLOv8(상태 검출) → BoxMOT(추적) → (선택) ST-GCN 이벤트 → Late-Fusion → 발송/로그
import os, time
import cv2
import numpy as np
from ultralytics import YOLO
from fusion_stabilizer import FallStabilizer
from notifier import Notifier

USE_STGCN = False 
MODEL_PATH = "/home/sung/runs/detect/train15/weights/best.pt"  # ← 자신의 best.pt 경로로 수정 가능 (또는 "yolov8s.pt")
SOURCE = 0         # 0=웹캠, 또는 "video.mp4"
IMG_SIZE = 896
CONF_THRES = 0.25
FALLEN_CLASS_IDX = 0  # data.yaml의 names에서 fallen_person의 인덱스
COOLDOWN_S = 8        # 같은 트랙 알림 쿨다운(초)
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
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter + 1e-9
    return inter / union


def fallen_conf_from_dets(track_xyxy, dets_np, fallen_cls_idx):
    """트랙 박스와 같은 프레임의 detections에서 fallen_person(cls=idx)을 IoU로 매칭해 conf를 가져옴"""
    if dets_np is None or len(dets_np) == 0:
        return 0.0
    mask = dets_np[:, 5].astype(int) == fallen_cls_idx
    cand = dets_np[mask]
    if cand.shape[0] == 0:
        return 0.0
    ious = np.array([iou_xyxy(track_xyxy, c[:4]) for c in cand], dtype=float)
    j = int(np.argmax(ious))
    if ious[j] < 0.1:
        return 0.0
    return float(cand[j, 4])


def main():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("alerts", exist_ok=True)
    method = "with_stgcn" if USE_STGCN else "no_stgcn"
    logf = open(f"logs/{method}.csv", "w", buffering=1)
    print("video_id,ts,tid,p_state,p_event,R,trigger", file=logf)

    # 1) 모델 로드
    model = YOLO(MODEL_PATH)

    # 2) 소스 열기 & FPS
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {SOURCE}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else FPS_FALLBACK

    # 3) 트래커
    try:
        tracker = _Tracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=fps)
    except TypeError:
        try:
            tracker = _Tracker(det_thresh=0.5, match_thresh=0.8, buffer_size=30, frame_rate=fps)
        except TypeError:
            tracker = _Tracker()

    # 4) 안정화/발송 모듈
    stab = FallStabilizer(tau_on=0.7, tau_off=0.4, min_frames=8, still_sec=1.0, ema=0.8, fps=fps)
    notifier = Notifier(webhook_url=None, cooldown_s=COOLDOWN_S)

    # 5) (옵션) ST-GCN
    stgcn = None
    if USE_STGCN:
        try:
            from events.stgcn_infer import STGCNInfer
            stgcn = STGCNInfer()  # 내부에서 트랙별 포즈 버퍼링 후 확률 반환
        except Exception as e:
            print(f"[WARN] ST-GCN unavailable ({e}); running without.")
            stgcn = None

    video_id = str(SOURCE)
    print(f"[INFO] Tracker: {TRACKER_NAME}, ST-GCN: {USE_STGCN}, Model: {MODEL_PATH}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if ts == 0 or np.isnan(ts):
            ts = time.time()  # 웹캠일 때 0이면 현재시간으로 대체

        # --- 검출 ---
        res = model(frame, imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)[0]
        boxes = res.boxes

        if boxes is None or len(boxes) == 0:
            tracks = tracker.update(np.zeros((0, 6), dtype=float), frame)
            cv2.imshow("demo", frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
            continue

        dets_np = np.concatenate(
            [boxes.xyxy.cpu().numpy(),
             boxes.conf.cpu().numpy()[:, None],
             boxes.cls.cpu().numpy()[:, None]],
            axis=1
        )  # M x [x1,y1,x2,y2,conf,cls]

        # --- 추적 ---
        tracks = tracker.update(dets_np, frame)  # K x [x1,y1,x2,y2,id,conf,cls,ind]

        # --- 트랙별 처리 ---
        for x1, y1, x2, y2, tid, conf, cls, _ in tracks:
            xyxy = (float(x1), float(y1), float(x2), float(y2))

            # p_state: 현재 프레임에서 fallen_person과 IoU 매칭된 박스의 conf
            p_state = fallen_conf_from_dets(xyxy, dets_np, FALLEN_CLASS_IDX)

            # p_event: ST-GCN (옵션)
            p_event = 0.0
            if USE_STGCN and stgcn is not None:
                try:
                    p_event = float(stgcn.push_and_prob(int(tid), frame, xyxy) or 0.0)
                except Exception:
                    p_event = 0.0

            # 안정화/결정
            trigger, R, still = stab.update(int(tid), p_state, p_event, xyxy)

            # 시각화
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            txt = f"ID {int(tid)} R={R:.2f} S={p_state:.2f} E={p_event:.2f}"
            cv2.putText(frame, txt, (int(x1), max(15, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if trigger:
                cv2.putText(frame, "FALL ALERT", (int(x1), int(y1) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                notifier.send(frame, int(tid), p_state, p_event, R, xyxy, ts,
                              wh=(frame.shape[1], frame.shape[0]), note=f"still={still}")

            # 로그 (전/후 비교용)
            print(f"{video_id},{ts:.3f},{int(tid)},{p_state:.3f},{p_event:.3f},{R:.3f},{int(trigger)}", file=logf)

        cv2.imshow("demo", frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):  # start of fall
            with open("logs/markers.csv", "a") as f:
                f.write(f"{app_run.SOURCE},{ts:.3f},start\n")
            print("[GT] start @", ts)
        elif key == ord('e'):  # end of fall
            with open("logs/markers.csv", "a") as f:
                f.write(f"{app_run.SOURCE},{ts:.3f},end\n")
            print("[GT] end @", ts)
        elif key == ord('n'):  # non-fall notable action (헷갈리는 동작)
            with open("logs/markers.csv", "a") as f:
                f.write(f"{app_run.SOURCE},{ts:.3f},nonfall\n")
            print("[GT] nonfall @", ts)

    logf.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
