# Model partitioning toolchain

These tools turn a standard HuggingFace causal LM into **browser-ready shards** —
a FRONT shard (embedding + early layers), one or more interior shards, and a BACK
shard (late layers + head) — that Osiris Compute streams across a circle of devices
and runs with `onnxruntime-web` (WebGPU).

Only a few **kilobytes of hidden-state** travel device→device per token; the weights
never move once placed. That is what lets a model too big for any single device run
across a laptop plus a handful of phones.

## The pipeline

```
HuggingFace model
   │  optimum  main_export  (fp32 ONNX, with-past / KV cache)
   ▼
model.onnx (fp32)
   │  onnxruntime  MatMulNBitsQuantizer  (weights → int4, activations stay fp32 = WebGPU-safe)
   ▼
model_q4.onnx
   │  osiris_partition.py  (seam-aware slice into FRONT / interior×K / BACK)
   ▼
qFRONT.onnx  qMID0..N.onnx  qBACK.onnx  tokenizer.json   → serve as static files
```

### 1. Export + quantize (`export_box.sh`)

Run on any box with enough RAM for the export step (a 7B fp32 export wants ~64GB
incl. swap; smaller models need much less). It installs deps into a venv, exports
the model to fp32 ONNX with KV-cache, quantizes the MatMul weights to int4, then
calls the partitioner.

```bash
HF_TOKEN=hf_xxx SWAP_GB=64 ./export_box.sh <id> <hf_repo>
# e.g. HF_TOKEN=hf_xxx ./export_box.sh qwen3b onnx-community/Qwen2.5-Coder-3B-Instruct
```

Outputs land in `work/<id>/`: the shard `.onnx` files, `tokenizer.json`, and a
`registry_entry.txt` with the client registry block to paste in.

> **Why int4 weights but fp32 activations?** WebGPU in the browser is happiest with
> fp32 activations; quantizing only the MatMul weights keeps the model small enough
> to ship to phones while staying numerically safe in `onnxruntime-web`.

### 2. Partition (`osiris_partition.py`)

Config-driven, K-way interior split (one interior shard per peer/phone). It:

- auto-detects layer count and the RMSNorm residual-stream **seam** tensors,
- plans a budget-aware split (per-shard byte ceiling so each shard loads in `ort-web`),
- splits the interior into `interior_stages` near-even shards (or `"auto"` from the
  phone byte budget),
- strips the dead-embedding shape-leak from interior + BACK shards (keeps them phone-sized),
- applies the embedding quant policy (`fp16` | `int8`) on FRONT,
- optionally bakes an fp16 head on BACK (`bake_head:false` for untied/int4 heads, e.g. Mistral/Llama),
- emits external-data ONNX for shards over the 2GB protobuf limit,
- **CPU-verifies** the chain end-to-end (FRONT → MID0 → … → BACK still answers),
- prints the client `MODELS` registry entry and `<option>`.

```bash
python3 osiris_partition.py --config configs/qwen3b.json --workdir work/qwen3b [--plan-only] [--skip-download] [--no-verify]
```

`export_box.sh` calls this for you; run it standalone to re-slice an
already-exported `model_q4.onnx` (e.g. to change the interior shard count).

## Config fields (`configs/*.json`)

| field | meaning |
|---|---|
| `id` / `name` | model id (URL path + registry key) and display name |
| `hf_repo` / `source_repo` | HuggingFace id to download / self-export from |
| `onnx_file` | path to the quantized ONNX inside the workdir (default `onnx/model_q4.onnx`) |
| `num_layers` | transformer block count (auto-detected if omitted) |
| `num_kv_heads` / `head_dim` | GQA KV-head count and per-head dim (for KV-cache wiring) |
| `vocab` / `hidden` | vocab size and hidden dim (for split byte planning) |
| `embed_policy` | `fp16` or `int8` embedding on FRONT |
| `bake_head` | `false` for untied / already-int4 lm_head (Mistral, Llama) |
| `mem_pattern` | emit `memPattern:false` (GQA decode-shape safety) — usually `false` |
| `interior_stages` | interior shard count `K`, or `"auto"` (derive from `phone_shard_mb`) |
| `max_shard_mb` / `phone_shard_mb` | per-shard load budgets for anchors vs phones |
| `chat` / `chat_template` / `stop` | chat formatting and stop token(s) |
| `default_prompt` / `def_temp` / `ep` | demo prompt, sampling temperature, ORT execution providers |

The included configs (`qwen15`, `qwen3b`, `mistral7b`, `llama3b`, `gemma2-2b`,
`qwen14b`) are working examples across several architectures.

## Serving the shards

Copy the resulting files to `public/models/<id>/` on whatever host serves the
client, so they are reachable at `/models/<id>/qFRONT.onnx`, etc. The client's
model registry (in `public/index.html`) points each model at `base:"/models/<id>/"`.
Range requests should be enabled so the browser can stream large shards.
