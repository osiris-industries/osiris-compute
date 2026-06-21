#!/usr/bin/env bash
# Fetch a small demo model's browser-ready shards into public/models/ so a fresh
# clone can run distributed inference immediately. Default: qwen (Qwen2.5-0.5B,
# ~1.3GB across FRONT / MID / BACK). Shards are NOT committed to git (too big);
# they are pulled from the public grid host.
#
#   ./fetch-demo-model.sh            # fetch the default demo model (qwen)
#   ./fetch-demo-model.sh qwen3b     # fetch a different served model
#   ./fetch-demo-model.sh qwen15dist # 1.5B in 6 interior shards — spreads across up to 6 phones
#
# Source host can be overridden:  MODEL_HOST=https://your.host ./fetch-demo-model.sh
set -euo pipefail

ID="${1:-qwen}"
HOST="${MODEL_HOST:-https://compute.osirisindustries.net}"
DIR="$(cd "$(dirname "$0")" && pwd)/public/models/$ID"

echo "Fetching '$ID' shards from $HOST/models/$ID/ -> $DIR"
mkdir -p "$DIR"

# Discover the shard file list from the live registry would be ideal; for the
# demo we fetch the known shard names and skip any that 404 (model layouts vary:
# single qMID vs qMID0..N).
files=(qFRONT.onnx qFRONT.onnx.data qBACK.onnx qBACK.onnx.data tokenizer.json
       qMID.onnx qMID.onnx.data
       qMID0.onnx qMID1.onnx qMID2.onnx qMID3.onnx qMID4.onnx qMID5.onnx)

got=0
for f in "${files[@]}"; do
  url="$HOST/models/$ID/$f"
  code=$(curl -s -o /dev/null -w '%{http_code}' -I "$url" || echo 000)
  if [ "$code" = "200" ]; then
    echo "  -> $f"
    curl -fL -sS --retry 5 --retry-delay 2 --retry-all-errors -C - -o "$DIR/$f" "$url"
    got=$((got+1))
  fi
done

if [ "$got" -eq 0 ]; then
  echo "No shards found for '$ID' at $HOST/models/$ID/ — check the model id." >&2
  exit 1
fi

echo "Done. $got files in $DIR"
echo "Now run:  node server.js   and pick '$ID' in the model dropdown."
