#!/usr/bin/env python3
"""Osiris generalized LLM partitioner — config-driven FRONT / N-interior / BACK
sharding for the browser compute grid. Consolidates every lesson from the qwen15 +
qwen3b landings, now with **K-way interior** (one shard per peer/phone):

  * auto seam detection (RMSNorm residual stream)
  * budget-aware split planning (per-shard byte ceiling for ort-web load)
  * K-way interior split: interior_stages = N (or "auto" from phone budget) ->
    qMID0..qMID{N-1} + a midStages[] registry block (back-compat: N=1 => single qMID)
  * dead-embedding shape-leak strip (every interior + BACK)  -> phone-sized shards
  * embedding quant policy (fp16 | int8) on FRONT             -> fits the load ceiling
  * fp16 head bake on BACK (skippable: untied/already-int4 heads via bake_head:false)
  * external-data ONNX (3B+ ship .onnx + .onnx_data)
  * GQA-safe: mem_pattern:false emitted; value_info cleared per slice
  * CPU verify (FRONT -> MID0 -> ... -> BACK still answers)
  * emits the client MODELS registry entry (+ <option>)

Usage:
  python osiris_partition.py --config configs/mistral7b.json [--plan-only]
                             [--skip-download] [--no-verify]
Run detached (systemd-run / setsid) on the build box per the CLAUDE.md rule.
"""
import os, sys, gc, json, re, argparse, numpy as np
import onnx
from onnx import numpy_helper, helper, TensorProto

# ---------------- config ----------------
def load_cfg(path):
    cfg = json.load(open(path))
    cfg.setdefault("onnx_file", "onnx/model_q4.onnx")
    cfg.setdefault("embed_policy", "fp16")      # fp16 | int8
    cfg.setdefault("max_shard_mb", 1100)        # ort-web per-shard load budget (anchors)
    cfg.setdefault("phone_shard_mb", 600)
    cfg.setdefault("interior_stages", 1)        # int K, or "auto" (derive K from phone budget)
    cfg.setdefault("bake_head", True)           # False for untied/already-int4 lm_head (Llama/Mistral)
    cfg.setdefault("mem_pattern", False)        # emit memPattern:false (GQA decode-shape safety)
    cfg.setdefault("embed_name", "model.embed_tokens.weight")
    cfg.setdefault("seam_template", "/model/layers.{lm1}/Add_1_output_0")
    cfg.setdefault("layer_regex", r"layers\.(\d+)")
    cfg.setdefault("ep", ["webgpu"])
    cfg.setdefault("def_temp", 0.5)
    cfg.setdefault("chat_template", None)        # "...{user}..." ; None => Qwen default
    return cfg

def mb(n): return n / (1 << 20)
PROTO_LIMIT = 1_900_000_000   # protobuf single-file ceiling (2GB) with margin

def save_model(m, path):
    tot = sum(len(i.raw_data) for i in m.graph.initializer)
    dpath = path + ".data"
    if os.path.exists(dpath):
        try: os.remove(dpath)
        except OSError: pass
    if tot > PROTO_LIMIT:
        onnx.save(m, path, save_as_external_data=True, all_tensors_to_one_file=True,
                  location=os.path.basename(path) + ".data", size_threshold=1024, convert_attribute=False)
    else:
        onnx.save(m, path)

# ---------------- download ----------------
def ensure_model(cfg, workdir):
    onnx_path = os.path.join(workdir, cfg["onnx_file"])
    tok_path  = os.path.join(workdir, "tokenizer.json")
    if os.path.exists(onnx_path) and os.path.exists(tok_path):
        return onnx_path, tok_path
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    repo = cfg["hf_repo"]; print(f"downloading {repo} ...", flush=True)
    files = [cfg["onnx_file"], cfg["onnx_file"] + "_data", "tokenizer.json", "config.json", "generation_config.json"]
    for f in files:
        try: hf_hub_download(repo, f, local_dir=workdir)
        except Exception as e: print(f"  (skip {f}: {e})", flush=True)
    return onnx_path, tok_path

# ---------------- arch + seams ----------------
def detect_arch(m, cfg):
    outs = {o for n in m.graph.node for o in n.output}
    rx = re.compile(cfg["layer_regex"]); lyr = set()
    for n in m.graph.node:
        for s in (n.name, *n.output):
            mm = rx.search(s)
            if mm: lyr.add(int(mm.group(1)))
    n_layers = (max(lyr) + 1) if lyr else cfg.get("num_layers")
    def seam_into(k):
        cand = cfg["seam_template"].format(lm1=k - 1, k=k)
        if cand in outs: return cand
        for n in m.graph.node:
            if n.name.startswith(f"/model/layers.{k}/input_layernorm"):
                return n.input[0]
        raise SystemExit(f"seam into block {k} not found (tried {cand})")
    return n_layers, seam_into

def measure_bytes(m, cfg, n_layers):
    init_sz = {i.name: (len(i.raw_data) or int(np.prod(i.dims)) * 4) for i in m.graph.initializer}
    embed_b = init_sz.get(cfg["embed_name"], 0)
    rx = re.compile(cfg["layer_regex"]); consumers = {}
    for n in m.graph.node:
        mm = rx.search(n.name or "")
        if not mm: continue
        k = int(mm.group(1))
        for i in n.input:
            if i in init_sz: consumers.setdefault(i, set()).add(k)
    by_layer = {}
    for i, ks in consumers.items():
        if len(ks) == 1:
            k = next(iter(ks)); by_layer[k] = by_layer.get(k, 0) + init_sz[i]
    mids = sorted(by_layer)[n_layers // 4 : 3 * n_layers // 4] or sorted(by_layer)
    per_layer = int(np.median([by_layer[k] for k in mids])) if mids else 0
    return per_layer, embed_b

# ---------------- split planner ----------------
def plan_split(n_layers, per_layer_b, vocab, hidden, cfg):
    pol = cfg["embed_policy"]
    embed_b = vocab * hidden * (2 if pol == "fp16" else 1)
    head_b  = vocab * hidden * 2 if cfg.get("bake_head", True) else int(vocab * hidden * 0.6)  # int4 head ~0.6B/elt
    budget  = cfg["max_shard_mb"] * (1 << 20)
    # start with anchors as small as fit, give the rest to interior
    front_cap = max(1, int((budget - embed_b) / per_layer_b)) if per_layer_b else n_layers
    back_cap  = max(1, int((budget - head_b)  / per_layer_b)) if per_layer_b else n_layers
    front = min(front_cap, max(1, n_layers // 6))
    back  = min(back_cap,  max(1, n_layers // 6))
    F, M = front, n_layers - back
    if M <= F: M = F + 1
    return F, M, front, back, embed_b, head_b

def plan_interior(F, M, per_layer_b, cfg):
    """split interior [F:M] into K contiguous near-even stages."""
    Lint = M - F
    spec = cfg.get("interior_stages", 1)
    if spec == "auto":
        per = max(1, int(cfg["phone_shard_mb"] * (1 << 20) / per_layer_b)) if per_layer_b else Lint
        K = max(1, -(-Lint // per))            # ceil(Lint/per)
    else:
        K = max(1, min(int(spec), Lint))
    cuts = [F + (j * Lint) // K for j in range(K)] + [M]
    return K, cuts

# ---------------- registry ----------------
def kv(p, lo, hi): return sum(([f"{p}.{i}.key", f"{p}.{i}.value"] for i in range(lo, hi)), [])

def emit_registry(cfg, n_layers, F, M, K, cuts, seam_into):
    ct = json.dumps(cfg["chat_template"]) if cfg.get("chat_template") else "null"
    mp = "" if cfg.get("mem_pattern", False) is None else f", memPattern:{str(cfg.get('mem_pattern', False)).lower()}"
    tail = (f'chat:{str(cfg.get("chat", True)).lower()}, stop:{json.dumps(cfg.get("stop"))}, '
            f'chatTemplate:{ct}, defaultPrompt:{json.dumps(cfg.get("default_prompt","Hello"))}, '
            f'defTemp:{cfg["def_temp"]}, ep:{json.dumps(cfg["ep"])}{mp}')
    if K == 1:
        seamF, seamM = seam_into(F), seam_into(M)
        entry = (
f'''    {cfg["id"]}: {{ id:"{cfg["id"]}", name:"{cfg["name"]}", base:"/models/{cfg["id"]}/",
      files:{{front:"qFRONT.onnx", mid:"qMID.onnx", back:"qBACK.onnx", tok:"tokenizer.json"}},
      seamF:"{seamF}", seamM:"{seamM}",
      kvFront:[0,{F}], kvMid:[{F},{M}], kvBack:[{M},{n_layers}], kvHeads:{cfg["num_kv_heads"]}, headDim:{cfg["head_dim"]},
      {tail} }},''')
    else:
        stages = []
        for j in range(K):
            lo, hi = cuts[j], cuts[j + 1]
            stages.append(f'        {{file:"qMID{j}.onnx", kv:[{lo},{hi}], '
                          f'seamIn:"{seam_into(lo)}", seamOut:"{seam_into(hi)}"}}')
        mids = ",\n".join(stages)
        entry = (
f'''    {cfg["id"]}: {{ id:"{cfg["id"]}", name:"{cfg["name"]}", base:"/models/{cfg["id"]}/",
      files:{{front:"qFRONT.onnx", back:"qBACK.onnx", tok:"tokenizer.json"}},
      seamF:"{seam_into(F)}", seamM:"{seam_into(M)}",
      kvFront:[0,{F}], kvBack:[{M},{n_layers}], kvHeads:{cfg["num_kv_heads"]}, headDim:{cfg["head_dim"]},
      midStages:[
{mids}
      ],
      {tail} }},''')
    option = f'<option value="{cfg["id"]}">{cfg["name"]}</option>'
    return entry, option

# ---------------- fixes (unchanged logic) ----------------
def strip_embed_leak(path, lo, hi, seam, cfg):
    EMB = cfg["embed_name"]; m = onnx.load(path)
    g = next((n for n in m.graph.node if n.op_type == "Gather" and EMB in n.input), None)
    if g is not None:
        gout = g.output[0]
        for n in m.graph.node:
            for j, i in enumerate(n.input):
                if i == gout: n.input[j] = seam
    owned = set()
    for li in range(lo, hi):
        owned.add(f"past_key_values.{li}.key"); owned.add(f"past_key_values.{li}.value")
    local = f"past_key_values.{lo}.key" if hi > lo else None
    if local:
        for n in m.graph.node:
            for j, i in enumerate(n.input):
                if i.startswith("past_key_values.") and i not in owned: n.input[j] = local
    go = {o.name for o in m.graph.output}
    while True:
        consumed = set(go)
        for n in m.graph.node:
            for i in n.input: consumed.add(i)
        keep = [n for n in m.graph.node if any(o in consumed for o in n.output)]
        if len(keep) == len(m.graph.node): break
        m.graph.ClearField("node"); m.graph.node.extend(keep)
    used = set(i for n in m.graph.node for i in n.input)
    ki = [x for x in m.graph.initializer if x.name in used]
    m.graph.ClearField("initializer"); m.graph.initializer.extend(ki)
    kin = [x for x in m.graph.input if x.name in used]
    m.graph.ClearField("input"); m.graph.input.extend(kin)
    m.graph.ClearField("value_info"); save_model(m, path); del m; gc.collect()

def apply_front_policy(path, cfg):
    EMB = cfg["embed_name"]; pol = cfg["embed_policy"]; m = onnx.load(path)
    ei = next(i for i, x in enumerate(m.graph.initializer) if x.name == EMB)
    arr = numpy_helper.to_array(m.graph.initializer[ei])
    gnode = next(n for n in m.graph.node if n.op_type == "Gather" and EMB in n.input)
    gout = gnode.output[0]
    if pol == "fp16":
        m.graph.initializer[ei].CopyFrom(numpy_helper.from_array(arr.astype(np.float16), EMB))
        tmp = gout + "_f16"; gnode.output[0] = tmp
        m.graph.node.append(helper.make_node("Cast", [tmp], [gout], to=TensorProto.FLOAT, name="cast_embed_front_f32"))
    elif pol == "int8":
        s = (np.abs(arr).max(axis=1, keepdims=True) / 127.0).astype(np.float32); s[s == 0] = 1.0
        q = np.clip(np.round(arr / s), -127, 127).astype(np.int8)
        m.graph.initializer[ei].CopyFrom(numpy_helper.from_array(q, EMB))
        m.graph.initializer.append(numpy_helper.from_array(s, EMB + "_scale"))
        ids = gnode.input[1]; gnode.output[0] = gout + "_i8"
        m.graph.node.append(helper.make_node("Gather", [EMB + "_scale", ids], [gout + "_sc"], axis=0, name="gather_embed_scale"))
        m.graph.node.append(helper.make_node("Cast", [gout + "_i8"], [gout + "_f32"], to=TensorProto.FLOAT, name="cast_embed_i8_f32"))
        m.graph.node.append(helper.make_node("Mul", [gout + "_f32", gout + "_sc"], [gout], name="dequant_embed"))
    else:
        raise SystemExit(f"unknown embed_policy {pol}")
    m.graph.ClearField("value_info"); save_model(m, path); del m, arr; gc.collect()

def bake_fp16_head(path, cfg):
    EMB = cfg["embed_name"]; m = onnx.load(path)
    head = next(n for n in m.graph.node if "logits" in n.output)
    tnode = next((n for n in m.graph.node if n.op_type == "Transpose" and EMB in n.input), None)
    ei = next(i for i, x in enumerate(m.graph.initializer) if x.name == EMB)
    arr = numpy_helper.to_array(m.graph.initializer[ei]).astype(np.float16)
    if tnode is not None:
        perm = next((list(a.ints) for a in tnode.attribute if a.name == "perm"), None)
        wt = np.transpose(arr, perm) if perm else arr.T
        wname = EMB + "_transposed"
        m.graph.node.remove(tnode); del m.graph.initializer[ei]
        m.graph.initializer.append(numpy_helper.from_array(wt, wname))
        for j, i in enumerate(head.input):
            if i == tnode.output[0]: head.input[j] = wname
    else:
        m.graph.initializer[ei].CopyFrom(numpy_helper.from_array(arr, EMB))
    for j, i in enumerate(list(head.input)):
        ci = i + "_hf16"
        m.graph.node.append(helper.make_node("Cast", [i], [ci], to=TensorProto.FLOAT16, name=f"cast_head_in{j}"))
        head.input[j] = ci
    oj = [k for k, o in enumerate(head.output) if o == "logits"][0]; head.output[oj] = "logits_pre"
    m.graph.node.append(helper.make_node("Cast", ["logits_pre"], ["logits"], to=TensorProto.FLOAT, name="cast_logits_f32"))
    used = set(i for n in m.graph.node for i in n.input)
    keep = [x for x in m.graph.initializer if x.name in used]
    m.graph.ClearField("initializer"); m.graph.initializer.extend(keep)
    m.graph.ClearField("value_info"); save_model(m, path); del m, arr; gc.collect()

# ---------------- verify (K-way chain) ----------------
def verify(QD, tok_path, cfg, n_layers, F, M, K, cuts, seam_into):
    import onnxruntime as ort
    from tokenizers import Tokenizer
    KVH, HD = cfg["num_kv_heads"], cfg["head_dim"]
    stop = cfg.get("stop"); stops = set(stop if isinstance(stop, list) else ([stop] if stop is not None else []))
    tmpl = cfg.get("chat_template")
    so = ort.SessionOptions(); so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.enable_mem_pattern = False
    S = lambda p: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
    Fs = S(os.path.join(QD, "qFRONT.onnx"))
    Ms = [S(os.path.join(QD, f"qMID{j}.onnx")) for j in range(K)] if K > 1 else [S(os.path.join(QD, "qMID.onnx"))]
    Bs = S(os.path.join(QD, "qBACK.onnx"))
    tok = Tokenizer.from_file(tok_path); ek = lambda: np.zeros((1, KVH, 0, HD), np.float32)
    feed = lambda s, pool: {k: v for k, v in pool.items() if k in {i.name for i in s.get_inputs()}}
    def ids_for(u):
        if not cfg.get("chat", True): return tok.encode(u).ids
        if tmpl: return tok.encode(tmpl.replace("{user}", u)).ids
        return tok.encode("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n").ids
    def gen(u, maxnew=48):
        ids = ids_for(u)
        kvF = {k: ek() for k in kv("past_key_values", 0, F)}
        kvS = [{k: ek() for k in kv("past_key_values", cuts[j], cuts[j + 1])} for j in range(K)]
        kvB = {k: ek() for k in kv("past_key_values", M, n_layers)}
        out = []; nxt = None
        for step in range(len(ids) + maxnew):
            t = ids[step] if step < len(ids) else nxt
            fc = {"input_ids": np.array([[t]], np.int64), "attention_mask": np.ones((1, step + 1), np.int64), "position_ids": np.array([[step]], np.int64)}
            of = Fs.run(None, feed(Fs, {**fc, **kvF})); fd = dict(zip([x.name for x in Fs.get_outputs()], of))
            for k in fd:
                if k.startswith("present"): kvF[k.replace("present", "past_key_values")] = fd[k]
            sname = seam_into(F); carry = fd[sname]
            for j in range(K):
                s_in, s_out = seam_into(cuts[j]), seam_into(cuts[j + 1])
                om = Ms[j].run(None, feed(Ms[j], {s_in: carry, **fc, **kvS[j]})); md = dict(zip([x.name for x in Ms[j].get_outputs()], om))
                for k in md:
                    if k.startswith("present"): kvS[j][k.replace("present", "past_key_values")] = md[k]
                carry = md[s_out]
            ob = Bs.run(None, feed(Bs, {seam_into(M): carry, **fc, **kvB})); bd = dict(zip([x.name for x in Bs.get_outputs()], ob))
            for k in bd:
                if k.startswith("present"): kvB[k.replace("present", "past_key_values")] = bd[k]
            nxt = int(np.argmax(bd["logits"][0, -1]))
            if step >= len(ids) - 1:
                if nxt in stops: break
                out.append(nxt)
        return tok.decode(out)
    print(f"\n[verify  FRONT -> {K} interior -> BACK]", flush=True)
    for q in ["What is the capital of France?", "Name the first three planets from the sun."]:
        print("Q:", q, "\nA:", gen(q), "\n", flush=True)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--workdir", default=None)
    ap.add_argument("--plan-only", action="store_true"); ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    workdir = args.workdir or os.path.join(os.getcwd(), cfg["id"]); os.makedirs(workdir, exist_ok=True)
    if args.skip_download:
        onnx_path = os.path.join(workdir, cfg["onnx_file"]); tok_path = os.path.join(workdir, "tokenizer.json")
    else:
        onnx_path, tok_path = ensure_model(cfg, workdir)
    print(f"model: {onnx_path} ({mb(os.path.getsize(onnx_path)):.0f}MB graph)", flush=True)
    si = onnx_path[:-5] + ".si.onnx"
    if not os.path.exists(si):
        from onnx import shape_inference
        print("shape inference (disk) ...", flush=True)
        shape_inference.infer_shapes_path(onnx_path, si)
    onnx_path = si
    print("loading graph ...", flush=True); m = onnx.load(onnx_path)
    n_layers, seam_into = detect_arch(m, cfg)
    per_layer_b, embed_fp_b = measure_bytes(m, cfg, n_layers)
    vocab = cfg.get("vocab"); hidden = cfg.get("hidden")
    if not (vocab and hidden):
        for init in m.graph.initializer:
            if init.name == cfg["embed_name"]: vocab, hidden = int(init.dims[0]), int(init.dims[1]); break
    cfg["vocab"], cfg["hidden"] = vocab, hidden
    F, M, front, back, embed_b, head_b = plan_split(n_layers, per_layer_b, vocab, hidden, cfg)
    K, cuts = plan_interior(F, M, per_layer_b, cfg)
    print(f"arch: layers={n_layers} vocab={vocab} hidden={hidden} kvHeads={cfg['num_kv_heads']} headDim={cfg['head_dim']}", flush=True)
    print(f"sizes: per_layer~{mb(per_layer_b):.1f}MB embedding~{mb(embed_fp_b):.0f}MB", flush=True)
    print(f"\nplan (embed={cfg['embed_policy']}, bake_head={cfg.get('bake_head',True)}, budget={cfg['max_shard_mb']}MB, phone={cfg['phone_shard_mb']}MB):")
    print(f"  FRONT embed+[0:{F}] ({front}L) ~{mb(embed_b + front*per_layer_b):.0f}MB")
    print(f"  INTERIOR [{F}:{M}] -> K={K} stages, cuts={cuts}:")
    for j in range(K):
        lo, hi = cuts[j], cuts[j+1]
        print(f"     qMID{j}: layers[{lo}:{hi}] ({hi-lo}L) ~{mb((hi-lo)*per_layer_b):.0f}MB  seam {seam_into(lo)} -> {seam_into(hi)}")
    print(f"  BACK [{M}:{n_layers}]+head ({back}L) ~{mb(head_b + back*per_layer_b):.0f}MB")
    maxmid = max((cuts[j+1]-cuts[j]) for j in range(K)) * per_layer_b
    if maxmid > cfg["phone_shard_mb"] * (1<<20):
        print(f"  ⚠ largest interior shard ~{mb(maxmid):.0f}MB > phone budget {cfg['phone_shard_mb']}MB — raise interior_stages")
    entry, option = emit_registry(cfg, n_layers, F, M, K, cuts, seam_into)
    print("\n--- MODELS entry ---\n" + entry + "\n--- option ---\n              " + option)
    open(os.path.join(workdir, "registry_entry.txt"), "w").write(entry + "\n\n" + option + "\n")
    if args.plan_only:
        print(f"\n[plan-only] wrote registry_entry.txt to {workdir}"); return
    print("\nbuilding extractor (single shape inference) ...", flush=True)
    ex = onnx.utils.Extractor(m); del m; gc.collect()
    common = ["input_ids", "attention_mask", "position_ids"]
    def slice_save(name, IN, OUT):
        g = ex.extract_model(IN, OUT); save_model(g, os.path.join(workdir, name)); del g; gc.collect()
    print("slicing ...", flush=True)
    slice_save("qFRONT.onnx", common + kv("past_key_values", 0, F), [seam_into(F)] + kv("present", 0, F))
    if K == 1:
        slice_save("qMID.onnx", [seam_into(F)] + common + kv("past_key_values", F, M), [seam_into(M)] + kv("present", F, M))
    else:
        for j in range(K):
            lo, hi = cuts[j], cuts[j + 1]
            slice_save(f"qMID{j}.onnx", [seam_into(lo)] + common + kv("past_key_values", lo, hi), [seam_into(hi)] + kv("present", lo, hi))
    slice_save("qBACK.onnx", [seam_into(M)] + common + kv("past_key_values", M, n_layers), ["logits"] + kv("present", M, n_layers))
    del ex; gc.collect()
    # strip leaks on every interior + back
    if K == 1:
        strip_embed_leak(os.path.join(workdir, "qMID.onnx"), F, M, seam_into(F), cfg)
    else:
        for j in range(K):
            lo, hi = cuts[j], cuts[j + 1]
            strip_embed_leak(os.path.join(workdir, f"qMID{j}.onnx"), lo, hi, seam_into(lo), cfg)
    strip_embed_leak(os.path.join(workdir, "qBACK.onnx"), M, n_layers, seam_into(M), cfg)
    apply_front_policy(os.path.join(workdir, "qFRONT.onnx"), cfg)
    if cfg.get("bake_head", True):
        bake_fp16_head(os.path.join(workdir, "qBACK.onnx"), cfg)
    else:
        print("  (skip bake_fp16_head — untied/already-int4 head)", flush=True)
    names = ["qFRONT.onnx"] + ([f"qMID{j}.onnx" for j in range(K)] if K > 1 else ["qMID.onnx"]) + ["qBACK.onnx"]
    for nm in names:
        p = os.path.join(workdir, nm); sz = mb(os.path.getsize(p)); ext = os.path.exists(p + ".data")
        print(f"  {nm}: {sz:.0f}MB{' (+external .data — too big for single-file client!)' if ext else ''}", flush=True)
    if not args.no_verify: verify(workdir, tok_path, cfg, n_layers, F, M, K, cuts, seam_into)
    print("=== PARTITION DONE ===", flush=True)

if __name__ == "__main__":
    main()
