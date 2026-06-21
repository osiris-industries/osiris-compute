#!/usr/bin/env python3
"""Osiris *sharded build* — export browser shards WITHOUT ever materializing the
whole fp32 model. A transformer is a stack of independent layers, so the BUILD
shards the same way inference does: export each contiguous stage
(FRONT = embed+layers[0:F], interior_j = layers[cuts[j]:cuts[j+1]],
BACK = layers[M:N]+norm+head) as a standalone ONNX that already satisfies the
Osiris shard I/O contract, then int4-quantize each stage on its own.

Each stage:
  inputs : <seam_in hidden>  (FRONT takes input_ids instead),
           input_ids, attention_mask, position_ids,
           past_key_values.{i}.key / .value   (global layer index i)
  outputs: <seam_out hidden> (BACK emits logits),
           present.{i}.key / .value

Stages recompute RoPE + causal mask locally from (attention_mask, position_ids),
which the contract passes to every stage — so each stage is fully self-contained
and no box ever holds more than its own layers.

The attention is reimplemented from the layer's OWN submodules (q/k/v/o_proj,
layernorms, mlp) so we don't depend on the transformers Cache/forward API, only
on stable weight modules. Matches Llama/Mistral/Qwen2 (RMSNorm + RoPE + GQA).

Phase 0 (this file): prove a stage chain reproduces the full model's logits on a
tiny model (numerical equivalence), then export+verify via onnxruntime. Lean
per-stage weight loading (so 14B fits a 16GB box) and the distributed scheduler
build on top of this exact-correctness core.

Usage:
  python osiris_build_sharded.py --repo <hf_id> --workdir out --front 2 --back 2 --interior 2 \
      [--equiv-only] [--seq 5]
"""
import os, sys, math, json, argparse, gc
import numpy as np


# ---------------- stage wrapper (torch) ----------------
def build_wrapper_classes():
    """Defined inside a fn so the module imports without torch present."""
    import torch, torch.nn as nn
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    NEG = None  # set per-dtype at call time

    def causal_additive_mask(attention_mask, T, P, dtype):
        """(b,1,T,P+T) additive mask: 0 allowed, big-negative disallowed.
        query t (current) is at key index P+t; attends keys 0..P+t (causal),
        and only where attention_mask==1 (padding)."""
        b = attention_mask.shape[0]; L = P + T
        neg = torch.finfo(dtype).min
        key_idx = torch.arange(L, device=attention_mask.device)
        q_pos = P + torch.arange(T, device=attention_mask.device)
        causal = (key_idx[None, :] <= q_pos[:, None])        # (T, L) bool
        pad = attention_mask[:, None, :].bool()              # (b,1,L)
        allow = causal[None, None, :, :] & pad[:, :, None, :]  # (b,1,T,L)
        m = torch.zeros(b, 1, T, L, dtype=dtype)
        m = m.masked_fill(~allow, neg)
        return m

    def run_layer(layer, h, mask4d, cos, sin, past_k, past_v, nH, nKV, hd):
        b, T, _ = h.shape
        residual = h
        x = layer.input_layernorm(h)
        sa = layer.self_attn
        q = sa.q_proj(x).view(b, T, nH, hd).transpose(1, 2)   # (b,nH,T,hd)
        k = sa.k_proj(x).view(b, T, nKV, hd).transpose(1, 2)  # (b,nKV,T,hd)
        v = sa.v_proj(x).view(b, T, nKV, hd).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = torch.cat([past_k, k], dim=2)                     # (b,nKV,P+T,hd)
        v = torch.cat([past_v, v], dim=2)
        present_k, present_v = k, v
        rep = nH // nKV
        kr, vr = repeat_kv(k, rep), repeat_kv(v, rep)
        attn = torch.matmul(q, kr.transpose(2, 3)) / math.sqrt(hd)
        attn = attn + mask4d
        attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        o = torch.matmul(attn, vr).transpose(1, 2).reshape(b, T, nH * hd)
        o = sa.o_proj(o)
        h = residual + o
        residual = h
        x = layer.post_attention_layernorm(h)
        h = residual + layer.mlp(x)
        return h, present_k, present_v

    class Stage(nn.Module):
        """A contiguous stage [lo:hi]. role in {front, interior, back}."""
        def __init__(self, model, lo, hi, role, nH, nKV, hd):
            super().__init__()
            self.lo, self.hi, self.role = lo, hi, role
            self.nH, self.nKV, self.hd = nH, nKV, hd
            self.layers = nn.ModuleList([model.model.layers[i] for i in range(lo, hi)])
            self.rotary = model.model.rotary_emb
            if role == "front":
                self.embed = model.model.embed_tokens
            if role == "back":
                self.norm = model.model.norm
                self.lm_head = model.lm_head

        def forward(self, x, attention_mask, position_ids, *past):
            import torch
            if self.role == "front":
                h = self.embed(x).float()
            else:
                h = x
            P = past[0].shape[2] if past else 0
            T = h.shape[1]
            cos, sin = self.rotary(h, position_ids)
            mask4d = causal_additive_mask(attention_mask, T, P, h.dtype)
            presents = []
            for li, layer in enumerate(self.layers):
                pk = past[2 * li] if past else h.new_zeros(h.shape[0], self.nKV, 0, self.hd)
                pv = past[2 * li + 1] if past else h.new_zeros(h.shape[0], self.nKV, 0, self.hd)
                h, ck, cv = run_layer(layer, h, mask4d, cos, sin, pk, pv,
                                      self.nH, self.nKV, self.hd)
                presents += [ck, cv]
            if self.role == "back":
                h = self.norm(h)
                h = self.lm_head(h)
            return tuple([h] + presents)

    return Stage


# ---------------- stage plan + names ----------------
def plan_stages(n_layers, front, back, interior):
    F, M = front, n_layers - back
    if M <= F:
        M = F + 1
    Lint = M - F
    K = max(1, min(interior, Lint))
    cuts = [F + (j * Lint) // K for j in range(K)] + [M]
    return F, M, K, cuts


def seam_name(k):
    # our own clean seam contract; chains because stage j out == stage j+1 in
    return f"hidden_at_{k}"


def kv_in(i):  return [f"past_key_values.{i}.key", f"past_key_values.{i}.value"]
def kv_out(i): return [f"present.{i}.key", f"present.{i}.value"]


# ---------------- export one stage to ONNX ----------------
def export_stage(Stage, model, lo, hi, role, nH, nKV, hd, n_layers, path, seq=4):
    import torch
    stage = Stage(model, lo, hi, role, nH, nKV, hd).eval()
    b, T, P = 1, seq, 0
    if role == "front":
        x = torch.randint(0, model.config.vocab_size, (b, T), dtype=torch.long)
        in0 = "input_ids"
    else:
        x = torch.randn(b, T, model.config.hidden_size)
        in0 = seam_name(lo)
    attn = torch.ones(b, T, dtype=torch.long)
    pos = torch.arange(T, dtype=torch.long).unsqueeze(0)
    pasts = []
    for i in range(lo, hi):
        pasts += [torch.zeros(b, nKV, P, hd), torch.zeros(b, nKV, P, hd)]
    inputs = tuple([x, attn, pos] + pasts)

    in_names = [in0, "attention_mask", "position_ids"]
    for i in range(lo, hi):
        in_names += kv_in(i)
    out0 = "logits" if role == "back" else seam_name(hi)
    out_names = [out0]
    for i in range(lo, hi):
        out_names += kv_out(i)

    dyn = {in0: {0: "b", 1: "T"}, "attention_mask": {0: "b", 1: "L"},
           "position_ids": {0: "b", 1: "T"}, out0: {0: "b", 1: "T"}}
    for i in range(lo, hi):
        for nm in kv_in(i):  dyn[nm] = {0: "b", 2: "P"}
        for nm in kv_out(i): dyn[nm] = {0: "b", 2: "L"}

    torch.onnx.export(stage, inputs, path, input_names=in_names, output_names=out_names,
                      dynamic_axes=dyn, opset_version=17, do_constant_folding=True, dynamo=False)
    del stage; gc.collect()
    return in_names, out_names


# ---------------- torch-level equivalence proof ----------------
def prove_equivalence(repo, front, back, interior, seq):
    import torch
    from transformers import AutoModelForCausalLM
    print(f"loading {repo} (full, fp32, CPU) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32)
    model.eval()
    cfg = model.config
    nH = cfg.num_attention_heads
    nKV = getattr(cfg, "num_key_value_heads", nH)
    hd = getattr(cfg, "head_dim", cfg.hidden_size // nH)
    n_layers = cfg.num_hidden_layers
    Stage = build_wrapper_classes()
    F, M, K, cuts = plan_stages(n_layers, front, back, interior)
    print(f"arch: layers={n_layers} nH={nH} nKV={nKV} hd={hd} vocab={cfg.vocab_size}")
    print(f"plan: FRONT[0:{F}]  interior K={K} cuts={cuts}  BACK[{M}:{n_layers}]")

    ids = torch.randint(0, cfg.vocab_size, (1, seq), dtype=torch.long)
    attn = torch.ones(1, seq, dtype=torch.long)
    pos = torch.arange(seq, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        ref = model(input_ids=ids, attention_mask=attn, position_ids=pos).logits

        # chain stages
        stages = [(0, F, "front")] + [(cuts[j], cuts[j + 1], "interior") for j in range(K)] + [(M, n_layers, "back")]
        carry = ids
        for (lo, hi, role) in stages:
            st = Stage(model, lo, hi, role, nH, nKV, hd).eval()
            past = []
            for _ in range(lo, hi):
                past += [torch.zeros(1, nKV, 0, hd), torch.zeros(1, nKV, 0, hd)]
            out = st(carry, attn, pos, *past)
            carry = out[0]
        chained = carry
    diff = (ref - chained).abs().max().item()
    rel = diff / (ref.abs().max().item() + 1e-9)
    print(f"\n[torch equivalence] max|Δlogits|={diff:.3e}  rel={rel:.3e}")
    ok = diff < 1e-3
    print("  PASS ✓ stage chain == full model" if ok else "  FAIL ✗")
    return ok, model, (nH, nKV, hd, n_layers, F, M, K, cuts)


# ---------------- onnx (ort) equivalence ----------------
def prove_onnx(model, dims, workdir, seq):
    import torch, onnxruntime as ort
    nH, nKV, hd, n_layers, F, M, K, cuts = dims
    Stage = build_wrapper_classes()
    os.makedirs(workdir, exist_ok=True)
    stages = [(0, F, "front", "qFRONT.onnx")] + \
             [(cuts[j], cuts[j + 1], "interior", f"qMID{j}.onnx") for j in range(K)] + \
             [(M, n_layers, "back", "qBACK.onnx")]
    for (lo, hi, role, fn) in stages:
        export_stage(Stage, model, lo, hi, role, nH, nKV, hd, n_layers,
                     os.path.join(workdir, fn), seq=seq)
        print(f"  exported {fn}  layers[{lo}:{hi}] ({role})", flush=True)

    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = {fn: ort.InferenceSession(os.path.join(workdir, fn), so,
            providers=["CPUExecutionProvider"]) for (_, _, _, fn) in stages}

    ids = np.random.randint(0, model.config.vocab_size, (1, seq)).astype(np.int64)
    attn = np.ones((1, seq), np.int64)
    pos = np.arange(seq, dtype=np.int64)[None, :]
    with torch.no_grad():
        ref = model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(attn),
                    position_ids=torch.tensor(pos)).logits.numpy()

    carry = ids
    for (lo, hi, role, fn) in stages:
        s = sess[fn]
        feed = {"attention_mask": attn, "position_ids": pos}
        feed["input_ids" if role == "front" else seam_name(lo)] = carry
        for i in range(lo, hi):
            feed[f"past_key_values.{i}.key"] = np.zeros((1, nKV, 0, hd), np.float32)
            feed[f"past_key_values.{i}.value"] = np.zeros((1, nKV, 0, hd), np.float32)
        # only feed inputs the graph declares
        want = {x.name for x in s.get_inputs()}
        feed = {k: v for k, v in feed.items() if k in want}
        out = s.run(None, feed)
        carry = out[0]
    diff = np.abs(ref - carry).max()
    rel = diff / (np.abs(ref).max() + 1e-9)
    print(f"\n[onnx/ort equivalence] max|Δlogits|={diff:.3e}  rel={rel:.3e}")
    ok = diff < 2e-2  # ort fp32 + opset math drift tolerance
    print("  PASS ✓ exported ONNX chain == full model" if ok else "  FAIL ✗")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--workdir", default="buildshard_out")
    ap.add_argument("--front", type=int, default=2)
    ap.add_argument("--back", type=int, default=2)
    ap.add_argument("--interior", type=int, default=2)
    ap.add_argument("--seq", type=int, default=5)
    ap.add_argument("--equiv-only", action="store_true")
    args = ap.parse_args()
    ok, model, dims = prove_equivalence(args.repo, args.front, args.back, args.interior, args.seq)
    if not ok:
        print("torch equivalence failed — not proceeding to ONNX"); sys.exit(1)
    if args.equiv_only:
        print("\n[equiv-only] done"); return
    ok2 = prove_onnx(model, dims, args.workdir, args.seq)
    print("\n=== BUILD-SHARD PROOF " + ("OK ===" if ok2 else "FAILED ==="))
    sys.exit(0 if ok2 else 1)


if __name__ == "__main__":
    main()
