# # events/stgcn_infer.py
# import math
# from collections import defaultdict, deque

# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
# from ultralytics import YOLO

# from .models.stgcn import STGCN  # ST-GCN 모델 (events/models/stgcn.py)

# def _xyxy_to_int(xyxy):
#     x1, y1, x2, y2 = xyxy
#     return int(x1), int(y1), int(x2), int(y2)

# def _normalize_kps(kps_xy, w, h):
#     # kps_xy: (J,2) in image coords -> normalize to [0,1] by frame size
#     out = kps_xy.copy()
#     out[:, 0] = np.clip(out[:, 0] / max(1, w), 0, 1)
#     out[:, 1] = np.clip(out[:, 1] / max(1, h), 0, 1)
#     return out

# class STGCNInfer:
#     """
#     추론용 모듈:
#       - YOLOv8 Pose로 프레임/박스에서 17개 관절 추출
#       - 트랙ID별로 시퀀스 버퍼링 후 ST-GCN으로 낙상 확률 산출
#     """
#     def __init__(
#         self,
#         ckpt: str = "events/ckpts/stgcn_fall.pt",
#         device: str = "cpu",
#         seq_len: int = 64,
#         min_buf: int = 24,
#         pose_model: str = "yolov8n-pose.pt",
#         imgsz: int = 896,
#         pose_conf: float = 0.25,
#     ):
#         self.ckpt = ckpt
#         self.seq_len = seq_len
#         self.min_buf = min_buf
#         self.imgsz = imgsz
#         self.pose_conf = pose_conf

#         # device
#         self.dev = torch.device(device)

#         # ST-GCN 로드
#         self.model = STGCN(num_class=2)
#         sd = torch.load(self.ckpt, map_location=self.dev)
#         if isinstance(sd, dict) and "model" in sd:
#             self.model.load_state_dict(sd["model"])
#         else:
#             self.model.load_state_dict(sd)
#         self.model.to(self.dev).eval()

#         # Pose 모델 (YOLOv8)
#         self.pose = YOLO(pose_model)

#         # 트랙별 (최근 seq_len 프레임) 버퍼
#         self.buf = defaultdict(lambda: deque(maxlen=self.seq_len))

#         print(f"[ST-GCN] loaded '{self.ckpt}' on {self.dev}; seq_len={seq_len}, min_buf={min_buf}")

#     def _pose_on_crop(self, frame, xyxy):
#         """xyxy로 crop → pose 추정 → (J,2) 반환(없으면 None)"""
#         h, w = frame.shape[:2]
#         x1, y1, x2, y2 = _xyxy_to_int(xyxy)
#         x1 = max(0, x1); y1 = max(0, y1); x2 = min(w - 1, x2); y2 = min(h - 1, y2)
#         if x2 <= x1 or y2 <= y1:
#             return None

#         crop = frame[y1:y2, x1:x2]
#         if crop.size == 0:
#             return None

#         # crop에서 포즈 추정
#         res = self.pose.predict(
#             crop, imgsz=self.imgsz, conf=self.pose_conf, verbose=False
#         )[0]

#         if res.keypoints is None or len(res.keypoints) == 0:
#             return None

#         # 가장 conf 높은 사람 1명 선택
#         # keypoints.xyn: (N, J, 2) normalized coords in crop
#         kpn = res.keypoints.xyn.cpu().numpy()  # 0~1
#         # bbox/score가 있으면 그걸로 정렬, 없으면 첫 번째
#         idx = 0
#         try:
#             if res.boxes is not None and len(res.boxes) > 0:
#                 scores = res.boxes.conf.cpu().numpy()
#                 idx = int(scores.argmax())
#         except Exception:
#             pass

#         kp_norm = kpn[idx]  # (J,2) in crop-normalized
#         # crop → 프레임 좌표로 변환
#         kp_xy = np.zeros_like(kp_norm)
#         kp_xy[:, 0] = kp_norm[:, 0] * (x2 - x1) + x1
#         kp_xy[:, 1] = kp_norm[:, 1] * (y2 - y1) + y1
#         return kp_xy  # (J,2) in frame coords

#     @torch.no_grad()
#     def push_and_prob(self, tid: int, frame: np.ndarray, xyxy) -> float:
#         """
#         한 프레임씩 호출:
#           - tid: 추적 ID (int)
#           - frame: BGR 이미지 (H,W,3)
#           - xyxy: 해당 tid의 박스 (x1,y1,x2,y2)
#         반환: 낙상(클래스1) 확률 [0..1]
#         """
#         h, w = frame.shape[:2]
#         kps = self._pose_on_crop(frame, xyxy)
#         if kps is None:
#             # 키포인트 실패 → 직전 값 유지, 확률 0
#             return 0.0

#         kps = _normalize_kps(kps, w, h)  # (J,2) in [0,1]
#         self.buf[int(tid)].append(kps.astype(np.float32))  # (J,2)

#         seq = self.buf[int(tid)]
#         if len(seq) < self.min_buf:
#             return 0.0

#         # 최근 seq_len 개로 구성 (부족하면 앞쪽을 패딩)
#         if len(seq) < self.seq_len:
#             pad = [seq[0]] * (self.seq_len - len(seq))
#             arr = np.stack(pad + list(seq), axis=0)  # (T,J,2)
#         else:
#             arr = np.stack(list(seq)[-self.seq_len :], axis=0)  # (T,J,2)

#         # (T,J,2) -> (1,C=2,T,J)
#         x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.dev)  # (1,2,T,V)
#         logits = self.model(x)  # (1,2)
#         prob = F.softmax(logits, dim=1)[0, 1].item()
#         return float(prob)


# events/stgcn_infer.py
# - YOLO Pose로 키포인트를 추출해 트랙별 버퍼에 쌓고
# - (가능하면) ST-GCN 체크포인트로 추론, 아니면 휴리스틱으로 확률 산출
# - 디버그 로그로 kpts/버퍼/확률을 확인할 수 있게 구성

from __future__ import annotations

import os
import math
import traceback
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
import torch
from ultralytics import YOLO


def _safe_to_numpy(x):
    try:
        return x.detach().cpu().numpy()
    except Exception:
        try:
            return np.array(x)
        except Exception:
            return None


class STGCNInfer:
    """
    Args:
        ckpt:           ST-GCN 체크포인트 경로 (TorchScript 또는 torch.load 가능 형태 가정)
        device:         "cpu" 또는 "cuda:0"
        seq_len:        시퀀스 입력 길이(T)
        min_buf:        prob을 내기 전 최소 버퍼 길이
        imgsz:          포즈 추론용 입력 크기
        pose_conf:      포즈 최소 conf
        pose_model_path:YOLO pose 가중치 경로 (명시 추천: 예) 'events/yolov8n-pose.pt')
        verbose:        디버그 로그 여부
    """

    def __init__(
        self,
        ckpt: str,
        device: str = "cpu",
        seq_len: int = 32,
        min_buf: int = 12,
        imgsz: int = 960,
        pose_conf: float = 0.10,
        pose_model_path: str = "yolov8n-pose.pt",
        verbose: bool = True,
    ):
        self.device = device
        self.seq_len = int(seq_len)
        self.min_buf = int(min_buf)
        self.imgsz = int(imgsz)
        self.pose_conf = float(pose_conf)
        self.verbose = bool(verbose)

        # 트랙별 포즈 버퍼 (tid -> list of (J,2) numpy)
        self.buf: Dict[int, List[np.ndarray]] = {}

        # 1) YOLO Pose 명시 경로로 로드
        self.pose_model = YOLO(pose_model_path)
        if self.verbose:
            print(f"[ST-GCN] pose model loaded from '{pose_model_path}' on {device}")

        # 2) ST-GCN 체크포인트 로드 (가능하면)
        self.model = None
        self.model_type = None  # "jit", "state_dict", None
        self._load_ckpt(ckpt)

        if self.verbose:
            print(f"[ST-GCN] loaded '{ckpt}' on {device}; seq_len={self.seq_len}, min_buf={self.min_buf}")

    # -------------------------------
    # 체크포인트 로드
    # -------------------------------
    def _load_ckpt(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            if self.verbose:
                print(f"[ST-GCN WARN] ckpt not found: {ckpt_path} -> using HEURISTIC fallback")
            return
        try:
            # TorchScript 우선
            self.model = torch.jit.load(ckpt_path, map_location=self.device)
            self.model.eval()
            self.model_type = "jit"
            if self.verbose:
                print("[ST-GCN] TorchScript checkpoint loaded")
            return
        except Exception:
            # state_dict 로드 시도
            try:
                ck = torch.load(ckpt_path, map_location=self.device)
                if isinstance(ck, dict) and "model" in ck and hasattr(ck["model"], "eval"):
                    # 이미 pickled nn.Module
                    self.model = ck["model"].to(self.device).eval()
                    self.model_type = "module"
                    if self.verbose:
                        print("[ST-GCN] nn.Module checkpoint loaded (pickled module)")
                elif isinstance(ck, dict):
                    # state_dict만 있는 경우: 아키텍처 모르면 instantiate 불가 → 포기
                    self.model = None
                    self.model_type = "state_dict"
                    if self.verbose:
                        print("[ST-GCN WARN] state_dict only; unknown architecture -> HEURISTIC fallback")
                else:
                    self.model = None
                    self.model_type = None
                    if self.verbose:
                        print("[ST-GCN WARN] unknown checkpoint format -> HEURISTIC fallback")
            except Exception as e:
                self.model = None
                self.model_type = None
                if self.verbose:
                    print(f"[ST-GCN WARN] failed to load ckpt: {e}\n{traceback.format_exc()}")

    # -------------------------------
    # 포즈 추출 (크롭 프레임 -> (J,2) 또는 None)
    # -------------------------------
    def _pose_from_crop(self, bgr_crop: np.ndarray) -> Optional[np.ndarray]:
        if bgr_crop is None or bgr_crop.size == 0:
            return None
        # RGB 변환
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        res = self.pose_model(rgb, imgsz=self.imgsz, conf=self.pose_conf, device=self.device, verbose=False)[0]
        kpts = getattr(res, "keypoints", None)
        if kpts is None:
            return None

        # 형태 표준화: (N_persons, J, 2)
        # ultralytics는 keypoints.xy 속성을 제공
        k_xy = getattr(kpts, "xy", None)
        if k_xy is not None:
            k = _safe_to_numpy(k_xy)
        else:
            k = _safe_to_numpy(kpts)
        if k is None:
            return None

        if k.ndim == 3:  # (N, J, 2) or (N, J, 3)
            if k.shape[-1] >= 2:
                person0 = k[0, :, :2]
            else:
                return None
        elif k.ndim == 2:
            # (J,2) 가정
            person0 = k[:, :2]
        else:
            return None

        return person0  # (J,2)

    # -------------------------------
    # 버퍼 → 입력 텐서 만들기
    # -------------------------------
    def _make_input_seq(self, seq_list: List[np.ndarray], H: int, W: int) -> torch.Tensor:
        """
        seq_list: list of (J,2) numpy, 길이 T' (T' >= 1)
        -> (1, T, J, 2) float tensor in [0,1] 로 정규화
        """
        T = min(len(seq_list), self.seq_len)
        seq = np.stack(seq_list[-T:], axis=0)  # (T, J, 2)
        # 화면 크기 정규화
        seq[..., 0] = np.clip(seq[..., 0] / max(1.0, W), 0.0, 1.0)
        seq[..., 1] = np.clip(seq[..., 1] / max(1.0, H), 0.0, 1.0)
        ten = torch.from_numpy(seq).float().unsqueeze(0)  # (1, T, J, 2)
        return ten.to(self.device)

    # -------------------------------
    # 휴리스틱 확률 (모델 없을 때 임시)
    # -------------------------------
    def _heuristic_prob(self, seq_list: List[np.ndarray]) -> float:
        """
        간단 휴리스틱:
         - 최근 8프레임 평균 속도(프레임 간 keypoint 이동량) 작고
         - 마지막 프레임에서 x-span > y-span (눕듯이 퍼짐)
         -> 넘어짐 확률↑
        """
        if len(seq_list) < 2:
            return 0.0
        tail = seq_list[-min(8, len(seq_list)):]  # 최근 few
        # 평균 속도
        v = []
        for a, b in zip(tail[:-1], tail[1:]):
            d = np.linalg.norm(b - a, axis=1).mean()
            v.append(d)
        mean_vel = float(np.mean(v)) if v else 0.0
        # 마지막 프레임 자세
        last = tail[-1]
        xspan = float(last[:, 0].max() - last[:, 0].min() + 1e-6)
        yspan = float(last[:, 1].max() - last[:, 1].min() + 1e-6)
        flatness = np.tanh((xspan / yspan) - 1.0)  # x>y면 양수
        stillness = np.exp(-mean_vel / (5.0))      # 속도 작을수록 1에 가까움
        p = max(0.0, min(1.0, 0.15 + 0.55 * stillness + 0.30 * flatness))
        return float(p)

    # -------------------------------
    # 실제 모델 추론 (있으면)
    # -------------------------------
    @torch.inference_mode()
    def _model_prob(self, inp: torch.Tensor) -> Optional[float]:
        """
        inp: (1, T, J, 2) torch float
        반환: 0~1 확률 또는 None(실패)
        """
        if self.model is None:
            return None
        try:
            out = self.model(inp)  # 모델에 따라 (N,C) 로짓 등
            out_np = _safe_to_numpy(out)
            if out_np is None:
                return None
            # (1,C) → 넘어짐 클래스가 1번이라고 가정하거나,
            # 이 부분은 ckpt에 따라 달라질 수 있음.
            # 안전하게 시그모이드/소프트맥스 둘 다 시도:
            if out_np.ndim == 2 and out_np.shape[0] == 1:
                logits = out_np[0]
                if logits.size == 1:
                    # binary logit
                    prob = 1.0 / (1.0 + math.exp(-float(logits[0])))
                else:
                    # multi-class → 마지막 또는 index=1을 넘어짐 가정
                    # 둘 다 계산해보고 큰 쪽 사용 (임시)
                    # softmax
                    ex = np.exp(logits - np.max(logits))
                    sm = ex / np.maximum(1e-9, ex.sum())
                    # 후보 1: index=1
                    p1 = float(sm[1]) if len(sm) > 1 else float(sm[-1])
                    # 후보 2: 마지막 인덱스
                    p2 = float(sm[-1])
                    prob = max(p1, p2)
            else:
                # 스칼라일 수도 있음
                val = float(np.ravel(out_np)[0])
                # 이미 확률일 수도 있고 로짓일 수도 있음 → 둘 다 커버
                if 0.0 <= val <= 1.0:
                    prob = val
                else:
                    prob = 1.0 / (1.0 + math.exp(-val))
            return max(0.0, min(1.0, float(prob)))
        except Exception:
            if self.verbose:
                print("[ST-GCN WARN] model forward failed:\n" + traceback.format_exc())
            return None

    # -------------------------------
    # 메인 API
    # -------------------------------
    def push_and_prob(self, tid: int, frame_bgr: np.ndarray, box) -> float:
        """
        tid:     트랙 ID
        frame:   BGR 프레임 (H,W,3)
        box:     (x1,y1,x2,y2) 절대좌표 또는 0~1 정규화 좌표
        return:  0~1 확률
        """
        H, W = frame_bgr.shape[:2]

        # 박스 정규화 처리 (둘 다 허용)
        if max(float(box[0]), float(box[1]), float(box[2]), float(box[3])) <= 1.0:
            x1 = float(box[0]) * W
            y1 = float(box[1]) * H
            x2 = float(box[2]) * W
            y2 = float(box[3]) * H
        else:
            x1, y1, x2, y2 = map(float, box)

        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(W - 1, int(x2)); y2 = min(H - 1, int(y2))
        if x2 <= x1 or y2 <= y1:
            if self.verbose:
                print(f"[STGCN-DBG] invalid crop {x1,y1,x2,y2}")
            # 버퍼 유지(공백 추가 없음)
            buf = self.buf.setdefault(int(tid), [])
            return 0.0

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            if self.verbose:
                print("[STGCN-DBG] empty crop")
            return 0.0

        # 1) 포즈 추출
        kpts = self._pose_from_crop(crop)
        if self.verbose:
            nk = 0 if kpts is None else kpts.shape[0]
            print(f"[STGCN-DBG] tid={tid} kpts_n={nk} crop={crop.shape}")

        # 2) 버퍼 업데이트 (키포인트가 있을 때만)
        buf = self.buf.setdefault(int(tid), [])
        if kpts is not None and kpts.size > 0:
            # 원본 좌표계로 보정
            kpts = kpts.copy()
            kpts[:, 0] += x1
            kpts[:, 1] += y1
            buf.append(kpts)
            if len(buf) > self.seq_len:
                buf.pop(0)

        if self.verbose:
            print(f"[STGCN-DBG] buf_len={len(buf)} (min_buf={self.min_buf}, seq_len={self.seq_len})")

        # 3) 최소 버퍼 미만이면 0
        if len(buf) < self.min_buf:
            return 0.0

        # 4) 입력 텐서 만들기
        inp = self._make_input_seq(buf, H, W)

        # 5) 모델 있으면 모델로, 없으면 휴리스틱
        prob = None
        if self.model is not None:
            prob = self._model_prob(inp)
        if prob is None:
            prob = self._heuristic_prob(buf)

        prob = float(max(0.0, min(1.0, prob)))
        if self.verbose:
            print(f"[STGCN-DBG] prob={prob:.3f}")
        return prob
