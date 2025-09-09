import os, time, json, cv2
try:
    import requests  # 있으면 웹훅 전송, 없어도 파일 로그는 동작
except Exception:
    requests = None

class Notifier:
    """
    trigger 시 경보(확률/위치/썸네일)를 웹훅 또는 파일로 기록
    """
    def __init__(self, webhook_url=None, cooldown_s=10):
        os.makedirs("alerts", exist_ok=True)
        self.webhook_url = webhook_url
        self.cooldown = cooldown_s
        self.last_sent = {}  # track_id -> last_ts

    def _crop_jpeg(self, frame, xyxy, path):
        x1, y1, x2, y2 = map(int, xyxy)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
        cv2.imwrite(path, frame[y1:y2, x1:x2])
        return path

    def send(self, frame, track_id, p_state, p_event, R, xyxy, ts, wh, note=""):
        now = time.time()
        # 스팸 방지(같은 트랙 쿨다운)
        if now - self.last_sent.get(track_id, 0) < self.cooldown:
            return False
        self.last_sent[track_id] = now

        W, H = wh
        cx = (xyxy[0] + xyxy[2]) / 2
        cy = (xyxy[1] + xyxy[3]) / 2

        thumb = f"alerts/track{track_id}_{int(now)}.jpg"
        self._crop_jpeg(frame, xyxy, thumb)

        payload = {
            "event": "fallen_alert",
            "track_id": int(track_id),
            "timestamp": float(ts),
            "probs": {"state": float(p_state), "event": float(p_event), "fused": float(R)},
            "bbox_xyxy": [float(x) for x in xyxy],
            "location": {
                "cx_px": float(cx), "cy_px": float(cy),
                "cx_norm": float(cx / W), "cy_norm": float(cy / H)
            },
            "thumb_path": thumb,
            "note": note
        }

        # 웹훅 전송(있으면), 실패하든 말든 파일 로그는 남김
        if self.webhook_url and requests:
            try:
                requests.post(self.webhook_url, json=payload, timeout=2)
            except Exception as e:
                print("webhook error:", e)

        with open("alerts/alerts.jsonl", "a") as f:
            f.write(json.dumps(payload) + "\n")
        return True
