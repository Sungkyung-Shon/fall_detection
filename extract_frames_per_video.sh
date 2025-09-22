set -euo pipefail

SRC="datasets/gmdcsa24"
DST="datasets/gmdcsa24_frames"
find "$SRC" -type f \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.mkv' \) -print0 \
| while IFS= read -r -d '' v; do
  rel="${v#"$SRC"/}"                
  base="${rel%.*}"                  
  outdir="$DST/$base"               
  mkdir -p "$outdir"
  if compgen -G "$outdir/*.jpg" > /dev/null; then
    echo "skip: $base (frames exist)"
    continue
  fi
  echo "extract: $base"
  ffmpeg -hide_banner -loglevel error -nostdin -y -i "$v" -vf fps=10 "$outdir/%06d.jpg" || {
    echo "FAIL: $base" >&2
  }
done
