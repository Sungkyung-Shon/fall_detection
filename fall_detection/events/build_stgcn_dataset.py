import argparse, os, glob, numpy as np, cv2
from ultralytics import YOLO

def is_fall(path: str) -> bool:
    # 디렉터리 경로에 /Fall/ 포함되면 1, 아니면 0
    p = path.replace("\\", "/")
    return "/Fall/" in p

def collect_video_dirs(frames_root: str):
    # 프레임(JPG/PNG)이 들어있는 모든 디렉터리 수집
    video_dirs = []
    for root, _, files in os.walk(frames_root):
        if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
            video_dirs.append(root)
    video_dirs.sort()
    return video_dirs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", default="datasets/gmdcsa24_frames")
    ap.add_argument("--out_root",    default="datasets/gmdcsa24_skeleton")
    ap.add_argument("--pose_model",  default="yolov8n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--stride", type=int, default=2, help="프레임 샘플 간격")
    ap.add_argument("--device", default="auto", help="'cpu' 또는 CUDA 인덱스 문자열 예: '0'")
    ap.add_argument("--skip_if_exist", action="store_true", help="이미 seq.npz 있으면 건너뜀")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    model = YOLO(args.pose_model)

    video_dirs = collect_video_dirs(args.frames_root)
    print(f"[INFO] video dirs: {len(video_dirs)}")
    saved = 0

    for vdir in video_dirs:
        rel = os.path.relpath(vdir, args.frames_root)
        outdir = os.path.join(args.out_root, rel)
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "seq.npz")

        if args.skip_if_exist and os.path.exists(outpath):
            print(f"skip existing: {rel}")
            continue

        frames = sorted(glob.glob(os.path.join(vdir, "*.jpg")))
        if not frames:
            continue

        seq = []
        step = max(1, args.stride)

        for fp in frames[::step]:
            im = cv2.imread(fp)
            if im is None:
                # 이미지 로드 실패 시 제로 패드
                seq.append(np.zeros((17, 3), dtype=np.float32))
                continue

            # 포즈 추론
            r = model.predict(im, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            kpts = r.keypoints

            if kpts is None or kpts.data is None or len(kpts.data) == 0:
                # 사람 미검출 → 제로 패드
                seq.append(np.zeros((17, 3), dtype=np.float32))
                continue

            kp = kpts.data.cpu().numpy()  # [N, 17, 3]
            # 가장 신뢰도 높은 사람 1명 선택
            if r.boxes is not None and r.boxes.conf is not None and len(r.boxes.conf) > 0:
                conf = r.boxes.conf.cpu().numpy()
                j = int(np.argmax(conf))
            else:
                mean_scores = kp[..., 2].mean(axis=1)
                j = int(np.argmax(mean_scores))

            seq.append(kp[j].astype(np.float32))  # [17, 3]

        arr = np.stack(seq, axis=0)  # [T, 17, 3]
        label = 1 if is_fall(vdir) else 0
        np.savez_compressed(outpath, kpts=arr, label=np.int64(label), rel=rel)
        print(f"[OK] {rel} -> {outpath}  T={arr.shape[0]} label={label}")
        saved += 1

    print(f"[DONE] saved {saved} sequences to '{args.out_root}'")

if __name__ == "__main__":
    main()
