# events/stgcn_infer.py
import math
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from .models.stgcn import STGCN  # ST-GCN 모델 (events/models/stgcn.py)

def _xyxy_to_int(xyxy):
    x1, y1, x2, y2 = xyxy
    return int(x1), int(y1), int(x2), int(y2)

def _normalize_kps(kps_xy, w, h):
    # kps_xy: (J,2) in image coords -> normalize to [0,1] by frame size
    out = kps_xy.copy()
    out[:, 0] = np.clip(out[:, 0] / max(1, w), 0, 1)
    out[:, 1] = np.clip(out[:, 1] / max(1, h), 0, 1)
    return out

class STGCNInfer:
    """
    추론용 모듈:
      - YOLOv8 Pose로 프레임/박스에서 17개 관절 추출
      - 트랙ID별로 시퀀스 버퍼링 후 ST-GCN으로 낙상 확률 산출
    """
    def __init__(
        self,
        ckpt: str = "events/ckpts/stgcn_fall.pt",
        device: str = "cpu",
        seq_len: int = 64,
        min_buf: int = 24,
        pose_model: str = "yolov8n-pose.pt",
        imgsz: int = 896,
        pose_conf: float = 0.25,
    ):
        self.ckpt = ckpt
        self.seq_len = seq_len
        self.min_buf = min_buf
        self.imgsz = imgsz
        self.pose_conf = pose_conf

        # device
        self.dev = torch.device(device)

        # ST-GCN 로드
        self.model = STGCN(num_class=2)
        sd = torch.load(self.ckpt, map_location=self.dev)
        if isinstance(sd, dict) and "model" in sd:
            self.model.load_state_dict(sd["model"])
        else:
            self.model.load_state_dict(sd)
        self.model.to(self.dev).eval()

        # Pose 모델 (YOLOv8)
        self.pose = YOLO(pose_model)

        # 트랙별 (최근 seq_len 프레임) 버퍼
        self.buf = defaultdict(lambda: deque(maxlen=self.seq_len))

        print(f"[ST-GCN] loaded '{self.ckpt}' on {self.dev}; seq_len={seq_len}, min_buf={min_buf}")

    def _pose_on_crop(self, frame, xyxy):
        """xyxy로 crop → pose 추정 → (J,2) 반환(없으면 None)"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = _xyxy_to_int(xyxy)
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w - 1, x2); y2 = min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # crop에서 포즈 추정
        res = self.pose.predict(
            crop, imgsz=self.imgsz, conf=self.pose_conf, verbose=False
        )[0]

        if res.keypoints is None or len(res.keypoints) == 0:
            return None

        # 가장 conf 높은 사람 1명 선택
        # keypoints.xyn: (N, J, 2) normalized coords in crop
        kpn = res.keypoints.xyn.cpu().numpy()  # 0~1
        # bbox/score가 있으면 그걸로 정렬, 없으면 첫 번째
        idx = 0
        try:
            if res.boxes is not None and len(res.boxes) > 0:
                scores = res.boxes.conf.cpu().numpy()
                idx = int(scores.argmax())
        except Exception:
            pass

        kp_norm = kpn[idx]  # (J,2) in crop-normalized
        # crop → 프레임 좌표로 변환
        kp_xy = np.zeros_like(kp_norm)
        kp_xy[:, 0] = kp_norm[:, 0] * (x2 - x1) + x1
        kp_xy[:, 1] = kp_norm[:, 1] * (y2 - y1) + y1
        return kp_xy  # (J,2) in frame coords

    @torch.no_grad()
    def push_and_prob(self, tid: int, frame: np.ndarray, xyxy) -> float:
        """
        한 프레임씩 호출:
          - tid: 추적 ID (int)
          - frame: BGR 이미지 (H,W,3)
          - xyxy: 해당 tid의 박스 (x1,y1,x2,y2)
        반환: 낙상(클래스1) 확률 [0..1]
        """
        h, w = frame.shape[:2]
        kps = self._pose_on_crop(frame, xyxy)
        if kps is None:
            # 키포인트 실패 → 직전 값 유지, 확률 0
            return 0.0

        kps = _normalize_kps(kps, w, h)  # (J,2) in [0,1]
        self.buf[int(tid)].append(kps.astype(np.float32))  # (J,2)

        seq = self.buf[int(tid)]
        if len(seq) < self.min_buf:
            return 0.0

        # 최근 seq_len 개로 구성 (부족하면 앞쪽을 패딩)
        if len(seq) < self.seq_len:
            pad = [seq[0]] * (self.seq_len - len(seq))
            arr = np.stack(pad + list(seq), axis=0)  # (T,J,2)
        else:
            arr = np.stack(list(seq)[-self.seq_len :], axis=0)  # (T,J,2)

        # (T,J,2) -> (1,C=2,T,J)
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.dev)  # (1,2,T,V)
        logits = self.model(x)  # (1,2)
        prob = F.softmax(logits, dim=1)[0, 1].item()
        return float(prob)
