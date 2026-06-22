#!/bin/bash
# Runs ON the ephemeral DO box. Export -> int4 quant -> slice (osiris_partition.py).
# Inputs in /root: osiris_partition.py, <cfg>.json, HF_TOKEN env. Outputs: /root/work/<id>/{qFRONT,qMID*,qBACK}.onnx + tokenizer.json
set -uo pipefail
exec >/root/box.log 2>&1
ID="${1:?model id}"; REPO="${2:?hf repo}"
echo "[$(date -u +%T)] === swap (insurance vs OOM on 7B fp32 export) ==="
if ! swapon --show | grep -q /swapfile; then
  fallocate -l ${SWAP_GB:-64}G /swapfile && chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile && echo "64G swap on"
fi
free -h | head -3
echo "[$(date -u +%T)] === deps ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq python3-venv python3-pip >/dev/null
python3 -m venv /root/venv && . /root/venv/bin/activate
pip install -q --upgrade pip
pip install -q torch --index-url https://download.pytorch.org/whl/cpu || { echo "TORCH FAIL"; echo "=== BOX FAIL ==="; exit 1; }
pip install -q optimum optimum-onnx onnx onnxruntime onnx_ir "transformers" tokenizers huggingface_hub accelerate sentencepiece protobuf \
  || { echo "PIP FAIL"; echo "=== BOX FAIL ==="; exit 1; }
python3 -c "import optimum.exporters.onnx; print('exporters import OK')" || { echo "EXPORTERS IMPORT FAIL"; echo "=== BOX FAIL ==="; exit 1; }
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p /root/work/"$ID"/onnx
echo "[$(date -u +%T)] === EXPORT $REPO (fp32 onnx, with-past) via main_export ==="
python3 - "$REPO" "$ID" <<'PY' || { echo "EXPORT FAIL"; echo "=== BOX FAIL ==="; exit 1; }
import sys
from optimum.exporters.onnx import main_export
repo, ID = sys.argv[1], sys.argv[2]
main_export(repo, output=f"/root/work/{ID}/onnx_fp32/", task="text-generation-with-past", opset=14, no_post_process=True)
print("export done")
PY
cp /root/work/"$ID"/onnx_fp32/tokenizer.json /root/work/"$ID"/tokenizer.json 2>/dev/null || true
echo "[$(date -u +%T)] === QUANT int4 (weights only, fp32 activations) ==="
python3 - "$ID" <<'PY' || { echo "QUANT FAIL"; echo "=== BOX FAIL ==="; exit 1; }
import sys, onnx
ID=sys.argv[1]
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
m=onnx.load(f"/root/work/{ID}/onnx_fp32/model.onnx")
q=MatMulNBitsQuantizer(m, bits=4, block_size=32, is_symmetric=True)   # weights->int4; activations stay fp32 (WebGPU-safe)
q.process()
q.model.save_model_to_file(f"/root/work/{ID}/onnx/model_q4.onnx", use_external_data_format=True)
print("quant done")
PY
echo "[$(date -u +%T)] === SLICE (osiris_partition.py) ==="
cd /root/work && python3 /root/osiris_partition.py --config /root/"$ID".json --workdir /root/work/"$ID" --skip-download \
  || { echo "SLICE FAIL"; echo "=== BOX FAIL ==="; exit 1; }
echo "[$(date -u +%T)] === shard sizes ==="
ls -la /root/work/"$ID"/qFRONT.onnx /root/work/"$ID"/qMID*.onnx /root/work/"$ID"/qBACK.onnx 2>/dev/null
echo "=== BOX DONE ==="
