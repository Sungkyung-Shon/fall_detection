#!/usr/bin/env bash
set -e

SRC="datasets/gmdcsa24"
DST="datasets/gmdcsa24_frames"

find "$SRC" -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' \) -print0 \
| while IFS= read -r -d '' v; do
  rel="${v#"$SRC"/}"
  outdir="$DST/${rel%/*}"
  mkdir -p "$outdir"

  # 이미 프레임 있으면 건너뜀
  if compgen -G "$outdir/*.jpg" > /dev/null; then
    echo "skip: $rel (frames exist)"
    continue
  fi

  if ffmpeg -hide_banner -loglevel error -nostdin -y -i "$v" -vf fps=10 "$outdir/%06d.jpg"; then
    echo "ok  : $rel"
  else
    echo "$v" >> ffmpeg_failed.txt
    echo "FAIL: $rel"
  fi
done
