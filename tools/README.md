# Model partitioning toolchain

These tools turn a standard HuggingFace causal LM into **browser-ready shards** —
a FRONT shard (embedding + early layers), one or more interior shards, and a BACK
shard (late layers + head) — that Osiris Compute streams across a circle of devices
and runs with `onnxruntime-web` (WebGPU).

Only a few **kilobytes of hidden-state** travel device→device per token; the weights
never move once placed. That is what lets a model too big for any single device run
across a laptop plus a handful of phones.

> **Model support — be precise.** This is validated end-to-end on the **Qwen2.5 family**
> (0.5B / 1.5B / 3B / 14B), which is what the live demo ships. The seam logic assumes the
> Qwen2/Llama tensor layout and currently uses Qwen RoPE + EOS, so **Llama / Mistral / Gemma
> configs are starting-point templates, not turnkey** — they need RoPE, EOS-token, and
> tensor-name adaptation. The lean builder now **fails loudly** if a checkpoint's tensor
> names don't match, rather than silently emitting a broken shard.

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
- **CPU smoke-tests** the chain end-to-end (FRONT → MID0 → … → BACK runs and decodes coherent text — a sanity check that the wiring/seams are intact, **not** a full logit-equivalence proof; see "What \"verified\" means" below),
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

The included configs span several architectures, but only the **Qwen2.5** ones
(`qwen15`, `qwen3b`, `qwen14b`) are validated end-to-end; `mistral7b`, `llama3b`, and
`gemma2-2b` are **templates** needing the per-architecture adaptation noted at the top.

## Serving the shards

Copy the resulting files to `public/models/<id>/` on whatever host serves the
client, so they are reachable at `/models/<id>/qFRONT.onnx`, etc. The client's
model registry (in `public/index.html`) points each model at `base:"/models/<id>/"`.
Range requests should be enabled so the browser can stream large shards.

---

## Lean & distributed K-way builder (no big box needed)

`osiris_partition.py` above slices a *pre-exported monolithic* ONNX, which means
one machine must first hold the whole fp32 model. The newer builder avoids that
entirely — it exports each stage independently and never materializes the whole
model, so a 16GB box can build a 14B and a tiny box can build a 1.5B.

Files:
- `osiris_build_sharded.py` — the exact-correctness core: exports one contiguous
  stage (FRONT / interior_j / BACK) to ONNX with the shard I/O contract, and proves
  **numerical logit-equivalence** (chained shards vs the unsharded model) on a small
  model, on a **prefill** pass. This is the real correctness guarantee in the toolchain.
- `osiris_build_lean.py` — **the K-way partitioner.** Meta-device skeleton + partial
  safetensors load of only a stage's layers → export → int4-quant → emit the client
  registry block. `--interior K` controls how many interior shards (one per phone).
- `osiris_build_distributed.py` — farm the K stages across machines (`--hosts
  local | ssh:user@host:/dir`), round-robin, scp shards back. No merge step.

> **What "verified" means (so it isn't oversold):** `osiris_build_sharded.py` proves
> *numerical* equivalence (prefill logits match the unsharded model) on a small model.
> `osiris_build_lean.py` / `osiris_build_distributed.py` instead run a **greedy-generation
> smoke test** through the quantized shard chain: it confirms the chain loads, the seams
> line up, and it decodes coherent text — but it does **not** re-compare logits against
> the full model on the autoregressive decode path. Treat a clean lean build as "wired
> correctly + int4-coherent," and the Phase-0 `sharded` proof as the equivalence guarantee.

Build a model split into K interior shards on ONE box:
```bash
python osiris_build_lean.py --repo Qwen/Qwen2.5-1.5B-Instruct \
  --workdir out --front 2 --back 2 --interior 6 --embed-policy fp16 \
  --id qwen15dist --name "Qwen2.5-1.5B (6 shards)"
# -> out/qFRONT.onnx, qMID0..5.onnx, qBACK.onnx, tokenizer.json, registry_entry.txt
```

Or farm the same build across machines:
```bash
python osiris_build_distributed.py build --repo Qwen/Qwen2.5-1.5B-Instruct \
  --workdir out --front 2 --back 2 --interior 6 --embed-policy fp16 \
  --hosts ssh:user@boxA:/build ssh:user@boxB:/build --remote-venv --max-parallel 2
```

Then serve `out/` at `public/models/<id>/` and paste the printed `registry_entry.txt`
block into `public/index.html`'s `MODELS` map (+ a `<option>`). The number of
interior shards = the number of phones the model can spread across at inference.
