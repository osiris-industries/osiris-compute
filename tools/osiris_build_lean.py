#!/usr/bin/env python3
"""Osiris sharded build — LEAN loader (Phase 2) + embed policy + registry emit.

Builds the deployable shard set (qFRONT / qMID* / qBACK + tokenizer.json + a
client registry entry) for a HuggingFace model WITHOUT ever materializing the
whole model. Each stage:

  1. build a meta-device skeleton (zero memory) from config,
  2. load ONLY that stage's tensors from safetensors (partial, assign=True),
  3. export the stage to ONNX (proven exact in Phase 0 / osiris_build_sharded.py),
  4. (FRONT) apply embed policy fp16/int8 to shrink the embedding,
  5. int4-quantize the stage ONNX on its own,
  6. free everything.

Peak RAM = one stage's weights, not the whole model. Lets a 16GB box build a 14B,
and lets THIS 3.8GB box build a real 0.5B/1.5B end to end.

Usage:
  python osiris_build_lean.py --repo Qwen/Qwen2.5-1.5B-Instruct --workdir out \
     --front 2 --back 2 --interior 6 --embed-policy fp16 \
     --id qwen15lean --name "Qwen2.5-1.5B (lean-built)" [--no-quant] [--no-verify]
"""
import os, sys, json, gc, argparse
try:
    import resource   # Unix-only; absent on Windows
except ImportError:
    resource = None
import numpy as np

from osiris_build_sharded import (build_wrapper_classes, plan_stages,
                                  export_stage, seam_name, kv_in, kv_out)


def rss_gb():
    if resource is None:
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


# ---------------- lean weight loading ----------------
def needed_keys(lo, hi, role, tied):
    ks = []
    for i in range(lo, hi):
        p = f"model.layers.{i}."
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            ks += [p + f"self_attn.{proj}.weight", p + f"self_attn.{proj}.bias"]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            ks += [p + f"mlp.{proj}.weight"]
        ks += [p + "input_layernorm.weight", p + "post_attention_layernorm.weight"]
    if role == "front":
        ks += ["model.embed_tokens.weight"]
    if role == "back":
        ks += ["model.norm.weight", "lm_head.weight"]
        if tied:
            ks += ["model.embed_tokens.weight"]
    return ks


def resolve_safetensors(repo, workdir):
    from huggingface_hub import hf_hub_download
    cache = os.path.join(workdir, "_hfsrc"); os.makedirs(cache, exist_ok=True)
    for f in ("config.json", "tokenizer.json", "generation_config.json", "tokenizer_config.json"):
        try: hf_hub_download(repo, f, local_dir=cache)
        except Exception: pass
    idx = None
    try:
        ip = hf_hub_download(repo, "model.safetensors.index.json", local_dir=cache)
        idx = json.load(open(ip))["weight_map"]
    except Exception:
        idx = None
    keymap = {}
    if idx:
        for fn in sorted(set(idx.values())):
            hf_hub_download(repo, fn, local_dir=cache)
        for k, fn in idx.items():
            keymap[k] = os.path.join(cache, fn)
    else:
        sp = hf_hub_download(repo, "model.safetensors", local_dir=cache)
        from safetensors import safe_open
        with safe_open(sp, framework="pt") as f:
            for k in f.keys():
                keymap[k] = sp
    return cache, keymap


def lean_load_stage(config, keymap, lo, hi, role, tied, embed_fp16=False):
    import torch
    from transformers import AutoModelForCausalLM
    from safetensors import safe_open
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    model.eval()
    needed = needed_keys(lo, hi, role, tied)
    want = [k for k in needed if k in keymap]
    # strict=False + assign=True load nothing silently if tensor names don't match.
    # Fail loudly on missing WEIGHT tensors (biases are legitimately absent on Llama/Mistral).
    missing_w = [k for k in needed if k not in keymap and k.endswith('.weight')]
    if missing_w:
        raise RuntimeError(
            f"lean_load_stage: {len(missing_w)} expected weight tensors are not in the checkpoint "
            f"(e.g. {missing_w[:3]}). The tensor names don't match the Qwen2/Llama layout this builder "
            f"assumes. This builder is validated on the Qwen2.5 family; other architectures need adaptation "
            f"(tensor names, RoPE, EOS) before they will produce correct shards.")
    byfile = {}
    for k in want:
        byfile.setdefault(keymap[k], []).append(k)
    sd = {}
    for fn, keys in byfile.items():
        with safe_open(fn, framework="pt") as f:
            for k in keys:
                t = f.get_tensor(k)
                sd[k] = t.half() if (embed_fp16 and role == "front" and k == "model.embed_tokens.weight") else t.float()
    model.load_state_dict(sd, strict=False, assign=True)
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    model.model.rotary_emb = Qwen2RotaryEmbedding(config).to("cpu")
    if role == "back" and tied:
        model.lm_head.weight = model.model.embed_tokens.weight
    del sd; gc.collect()
    return model


# ---------------- FRONT embed policy (shrink the embedding; quant skips Gather) ----------------
def apply_embed_policy(path, policy):
    import onnx
    from onnx import numpy_helper, helper, TensorProto
    if policy not in ("fp16", "int8"):
        return
    m = onnx.load(path)
    inits = {i.name: i for i in m.graph.initializer}
    # embedding = the Gather whose data input is the 2D float initializer with the most rows
    gather, emb = None, None; best = -1
    for n in m.graph.node:
        if n.op_type == "Gather" and n.input and n.input[0] in inits:
            t = inits[n.input[0]]
            if len(t.dims) == 2 and t.dims[0] > best:
                best = t.dims[0]; gather = n; emb = n.input[0]
    if gather is None:
        return
    ei = next(i for i, x in enumerate(m.graph.initializer) if x.name == emb)
    arr = numpy_helper.to_array(m.graph.initializer[ei])
    gout = gather.output[0]
    if policy == "fp16":
        m.graph.initializer[ei].CopyFrom(numpy_helper.from_array(arr.astype(np.float16), emb))
        tmp = gout + "_f16"; gather.output[0] = tmp
        m.graph.node.append(helper.make_node("Cast", [tmp], [gout], to=TensorProto.FLOAT, name="cast_embed_f32"))
    else:  # int8 per-row symmetric
        s = (np.abs(arr).max(axis=1, keepdims=True) / 127.0).astype(np.float32); s[s == 0] = 1.0
        q = np.clip(np.round(arr / s), -127, 127).astype(np.int8)
        m.graph.initializer[ei].CopyFrom(numpy_helper.from_array(q, emb))
        m.graph.initializer.append(numpy_helper.from_array(s, emb + "_scale"))
        ids = gather.input[1]; gather.output[0] = gout + "_i8"
        m.graph.node.append(helper.make_node("Gather", [emb + "_scale", ids], [gout + "_sc"], axis=0, name="gather_embed_scale"))
        m.graph.node.append(helper.make_node("Cast", [gout + "_i8"], [gout + "_f32"], to=TensorProto.FLOAT, name="cast_embed_i8"))
        m.graph.node.append(helper.make_node("Mul", [gout + "_f32", gout + "_sc"], [gout], name="dequant_embed"))
    m.graph.ClearField("value_info"); onnx.save(m, path)
    del m, arr; gc.collect()


# ---------------- per-stage int4 quant ----------------
def quant_int4(path_in, path_out):
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    m = onnx.load(path_in)
    q = MatMulNBitsQuantizer(m, block_size=32, is_symmetric=True)
    q.process()
    tot = sum(len(t.raw_data) if t.raw_data else 0 for t in q.model.model.graph.initializer)
    q.model.save_model_to_file(path_out, use_external_data_format=(tot > 1_900_000_000))
    del m, q; gc.collect()


# ---------------- registry entry (client MODELS block) ----------------
def emit_registry(workdir, cfg):
    nKV, hd, n_layers = cfg["nKV"], cfg["hd"], cfg["n_layers"]
    F, M, K, cuts = cfg["F"], cfg["M"], cfg["K"], cfg["cuts"]
    mids = ",\n".join(
        f'        {{file:"qMID{j}.onnx", kv:[{cuts[j]},{cuts[j+1]}], '
        f'seamIn:"{seam_name(cuts[j])}", seamOut:"{seam_name(cuts[j+1])}"}}'
        for j in range(K))
    entry = f'''    {cfg["id"]}: {{ id:"{cfg["id"]}", name:"{cfg["name"]}", base:"/models/{cfg["id"]}/",
      files:{{front:"qFRONT.onnx", back:"qBACK.onnx", tok:"tokenizer.json"}},
      seamF:"{seam_name(F)}", seamM:"{seam_name(M)}",
      kvFront:[0,{F}], kvBack:[{M},{n_layers}], kvHeads:{nKV}, headDim:{hd},
      midStages:[
{mids}
      ],
      chat:true, stop:{cfg["stop"]}, chatTemplate:null,
      defaultPrompt:{json.dumps(cfg["default_prompt"])}, defTemp:{cfg["def_temp"]},
      ep:["webgpu"], memPattern:false }},'''
    option = f'<option value="{cfg["id"]}">{cfg["name"]}</option>'
    open(os.path.join(workdir, "registry_entry.txt"), "w").write(entry + "\n\n" + option + "\n")
    return entry, option


# ---------------- verify: chain the quantized shards, greedy generate ----------------
def verify_chain(workdir, repo_cache, dims, prompt, maxnew=40):
    import onnxruntime as ort
    from tokenizers import Tokenizer
    nH, nKV, hd, n_layers, F, M, K, cuts = dims
    so = ort.SessionOptions(); so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.enable_mem_pattern = False
    S = lambda p: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
    Fs = S(os.path.join(workdir, "qFRONT.onnx"))
    Ms = [S(os.path.join(workdir, f"qMID{j}.onnx")) for j in range(K)]
    Bs = S(os.path.join(workdir, "qBACK.onnx"))
    tok = Tokenizer.from_file(os.path.join(repo_cache, "tokenizer.json"))
    ek = lambda lo, hi: {f"past_key_values.{i}.{t}": np.zeros((1, nKV, 0, hd), np.float32)
                         for i in range(lo, hi) for t in ("key", "value")}
    feed = lambda s, pool: {k: v for k, v in pool.items() if k in {i.name for i in s.get_inputs()}}
    text = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    ids = tok.encode(text).ids
    kvF = ek(0, F); kvS = [ek(cuts[j], cuts[j + 1]) for j in range(K)]; kvB = ek(M, n_layers)
    out = []; nxt = None
    for step in range(len(ids) + maxnew):
        t = ids[step] if step < len(ids) else nxt
        fc = {"input_ids": np.array([[t]], np.int64),
              "attention_mask": np.ones((1, step + 1), np.int64),
              "position_ids": np.array([[step]], np.int64)}
        of = Fs.run(None, feed(Fs, {**fc, **kvF})); fd = dict(zip([x.name for x in Fs.get_outputs()], of))
        for k in fd:
            if k.startswith("present"): kvF[k.replace("present", "past_key_values")] = fd[k]
        carry = fd[seam_name(F)]
        for j in range(K):
            om = Ms[j].run(None, feed(Ms[j], {seam_name(cuts[j]): carry, **fc, **kvS[j]}))
            md = dict(zip([x.name for x in Ms[j].get_outputs()], om))
            for k in md:
                if k.startswith("present"): kvS[j][k.replace("present", "past_key_values")] = md[k]
            carry = md[seam_name(cuts[j + 1])]
        ob = Bs.run(None, feed(Bs, {seam_name(M): carry, **fc, **kvB}))
        bd = dict(zip([x.name for x in Bs.get_outputs()], ob))
        for k in bd:
            if k.startswith("present"): kvB[k.replace("present", "past_key_values")] = bd[k]
        nxt = int(np.argmax(bd["logits"][0, -1]))
        if step >= len(ids) - 1:
            if nxt in (151643, 151645): break
            out.append(nxt)
    return tok.decode(out)


def main():
    from transformers import AutoConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--workdir", default="lean_out")
    ap.add_argument("--front", type=int, default=2)
    ap.add_argument("--back", type=int, default=2)
    ap.add_argument("--interior", type=int, default=4)
    ap.add_argument("--seq", type=int, default=4)
    ap.add_argument("--embed-policy", default="fp16", choices=["fp16", "int8", "none"])
    ap.add_argument("--id", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--stop", type=int, default=151645)
    ap.add_argument("--def-temp", type=float, default=0.5)
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--prompt", default="What is the capital of France?")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    config = AutoConfig.from_pretrained(args.repo)
    tied = bool(getattr(config, "tie_word_embeddings", False))
    nH = config.num_attention_heads
    nKV = getattr(config, "num_key_value_heads", nH)
    hd = getattr(config, "head_dim", config.hidden_size // nH)
    n_layers = config.num_hidden_layers
    F, M, K, cuts = plan_stages(n_layers, args.front, args.back, args.interior)
    mid_id = args.id or args.repo.split("/")[-1].lower().replace(".", "")
    mid_name = args.name or args.repo.split("/")[-1]
    print(f"arch: layers={n_layers} nH={nH} nKV={nKV} hd={hd} vocab={config.vocab_size} tied={tied}")
    print(f"plan: FRONT[0:{F}]  interior K={K} cuts={cuts}  BACK[{M}:{n_layers}]  embed={args.embed_policy}", flush=True)

    print("resolving safetensors ...", flush=True)
    repo_cache, keymap = resolve_safetensors(args.repo, args.workdir)
    Stage = build_wrapper_classes()
    stages = [(0, F, "front", "FRONT")] + \
             [(cuts[j], cuts[j + 1], "interior", f"MID{j}") for j in range(K)] + \
             [(M, n_layers, "back", "BACK")]
    for (lo, hi, role, tag) in stages:
        model = lean_load_stage(config, keymap, lo, hi, role, tied, embed_fp16=(args.embed_policy == "fp16"))
        raw = os.path.join(args.workdir, f"_{tag}.fp32.onnx")
        export_stage(Stage, model, lo, hi, role, nH, nKV, hd, n_layers, raw, seq=args.seq)
        del model; gc.collect()
        if role == "front" and args.embed_policy == "int8":
            apply_embed_policy(raw, args.embed_policy)
        final = os.path.join(args.workdir, {"FRONT": "qFRONT.onnx", "BACK": "qBACK.onnx"}.get(tag, f"q{tag}.onnx"))
        if args.no_quant:
            os.replace(raw, final)
        else:
            quant_int4(raw, final)
            try: os.remove(raw)
            except OSError: pass
        szmb = os.path.getsize(final) / (1 << 20)
        print(f"  [{tag}] layers[{lo}:{hi}] -> {os.path.basename(final)} {szmb:.1f}MB   peakRSS={rss_gb():.2f}GB", flush=True)

    import shutil
    try: shutil.copy(os.path.join(repo_cache, "tokenizer.json"), os.path.join(args.workdir, "tokenizer.json"))
    except Exception: pass

    cfg = dict(id=mid_id, name=mid_name, nKV=nKV, hd=hd, n_layers=n_layers, F=F, M=M, K=K, cuts=cuts,
               stop=args.stop, default_prompt=args.prompt, def_temp=args.def_temp)
    entry, option = emit_registry(args.workdir, cfg)
    print(f"\nPEAK RSS for whole build: {rss_gb():.2f} GB", flush=True)
    print("\n--- registry entry (registry_entry.txt) ---\n" + entry, flush=True)

    if not args.no_verify:
        dims = (nH, nKV, hd, n_layers, F, M, K, cuts)
        print("\n[verify] greedy-generating through the quantized shard chain ...", flush=True)
        txt = verify_chain(args.workdir, repo_cache, dims, args.prompt)
        print("Q:", args.prompt); print("A:", txt)
    print("\n=== LEAN BUILD DONE ===")


if __name__ == "__main__":
    main()
