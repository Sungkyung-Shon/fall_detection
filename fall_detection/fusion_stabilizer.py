from collections import defaultdict, deque
import math

class FallStabilizer:
    """
    입력: tid(트랙ID), p_state(YOLO fallen 확률), p_event(ST-GCN 확률), bbox(x1,y1,x2,y2)
    출력: (trigger(True/False), R(결합확률), still(True/False))
    """
    def __init__(self, tau_on=0.7, tau_off=0.4, min_frames=8,
                 still_sec=1.0, ema=0.8, fps=30, speed_th=5.0):
        self.tau_on = tau_on
        self.tau_off = tau_off
        self.min_frames = min_frames
        self.still_frames = max(1, int(still_sec * fps))
        self.ema = ema
        self.speed_th = speed_th

        self.buf = defaultdict(lambda: {
            "pS": 0.0, "pE": 0.0,
            "S": deque(maxlen=120), "E": deque(maxlen=120), "V": deque(maxlen=120),
            "center": None, "on": False
        })

    def update(self, tid, p_state, p_event, bbox):
        b = self.buf[tid]

        # 중심좌표 & 속도(픽셀/프레임)
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        if b["center"] is None:
            speed = 0.0
        else:
            dx = cx - b["center"][0]
            dy = cy - b["center"][1]
            speed = math.hypot(dx, dy)
        b["center"] = (cx, cy)

        # EMA 평활화
        b["pS"] = self.ema * b["pS"] + (1 - self.ema) * float(p_state)
        b["pE"] = self.ema * b["pE"] + (1 - self.ema) * float(p_event)
        b["S"].append(b["pS"])
        b["E"].append(b["pE"])
        b["V"].append(speed)

        # 정지 판정(최근 still_frames 동안 속도<th)
        vv = list(b["V"])[-self.still_frames:]
        still = len(vv) == self.still_frames and all(v < self.speed_th for v in vv)

        # Late-Fusion
        R = 0.6 * b["pS"] + 0.4 * b["pE"]
        if not still:
            R *= 0.5  # 움직이면 경보 완화

        # 히스테리시스 + 최소 지속 프레임
        long_enough = len(b["S"]) >= self.min_frames and all(
            x > self.tau_off for x in list(b["S"])[-self.min_frames:]
        )
        trigger = (R > self.tau_on) and long_enough and still
        if trigger:
            b["on"] = True
        if b["on"] and R < self.tau_off:
            b["on"] = False

        return trigger, R, still
