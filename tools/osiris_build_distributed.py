#!/usr/bin/env python3
"""Osiris sharded build — DISTRIBUTED scheduler (Phase 3).

Each stage of a model is an INDEPENDENT build job that produces one final shard
(qFRONT / qMIDj / qBACK). The shards ARE the deployable artifacts, so there is NO
merge step: every worker drops a finished file into the shared output dir. The
build therefore farms across as many boxes as you can reach — the same way
inference farms across phones.

Roles:
  worker  — build exactly ONE stage (lean-load only its layers -> export -> int4).
  build   — scheduler: plan stages, dispatch each to a host (round-robin over
            --hosts), run them concurrently, scp shards back, emit registry, verify.

Hosts:
  local                      worker as a local subprocess
  ssh:user@HOST:/remote/dir  worker on a remote box (its venv at <dir>/.venv with
                             --remote-venv); shard scp'd back to the shared workdir.
"""
import os, sys, json, argparse, subprocess, shutil, time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["osiris_build_sharded.py", "osiris_build_lean.py", "osiris_build_distributed.py"]
VENV_PY = os.path.join(HERE, ".venv/bin/python")
SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]


def stage_list(n_layers, front, back, interior):
    from osiris_build_sharded import plan_stages
    F, M, K, cuts = plan_stages(n_layers, front, back, interior)
    stages = [(0, F, "front", "FRONT")] + \
             [(cuts[j], cuts[j + 1], "interior", f"MID{j}") for j in range(K)] + \
             [(M, n_layers, "back", "BACK")]
    return F, M, K, cuts, stages


def shard_filename(tag):
    return {"FRONT": "qFRONT.onnx", "BACK": "qBACK.onnx"}.get(tag, f"q{tag}.onnx")


def parse_ssh(host):
    # ssh:user@HOST:/remote/dir  (and optional -i key via OSIRIS_SSH_KEY env)
    spec = host[len("ssh:"):]
    conn, rdir = spec.split(":", 1)
    key = os.environ.get("OSIRIS_SSH_KEY")
    base = ["ssh"] + (["-i", key] if key else []) + SSH_OPTS + [conn]
    scp = ["scp"] + (["-i", key] if key else []) + SSH_OPTS
    return conn, rdir, base, scp


# ---------------- worker: build ONE stage ----------------
def run_worker(args):
    import gc
    from transformers import AutoConfig
    from osiris_build_sharded import build_wrapper_classes, export_stage
    from osiris_build_lean import (resolve_safetensors, lean_load_stage,
                                   apply_embed_policy, quant_int4, rss_gb)
    os.makedirs(args.workdir, exist_ok=True)
    config = AutoConfig.from_pretrained(args.repo)
    tied = bool(getattr(config, "tie_word_embeddings", False))
    nH = config.num_attention_heads
    nKV = getattr(config, "num_key_value_heads", nH)
    hd = getattr(config, "head_dim", config.hidden_size // nH)
    n_layers = config.num_hidden_layers
    _, _, _, _, stages = stage_list(n_layers, args.front, args.back, args.interior)
    sel = [s for s in stages if s[3] == args.tag]
    if not sel:
        sys.exit(f"unknown stage {args.tag}; have {[s[3] for s in stages]}")
    lo, hi, role, tag = sel[0]
    repo_cache, keymap = resolve_safetensors(args.repo, args.workdir)
    Stage = build_wrapper_classes()
    model = lean_load_stage(config, keymap, lo, hi, role, tied, embed_fp16=(args.embed_policy == "fp16"))
    raw = os.path.join(args.workdir, f"_{tag}.fp32.onnx")
    export_stage(Stage, model, lo, hi, role, nH, nKV, hd, n_layers, raw, seq=args.seq)
    del model; gc.collect()
    if role == "front" and args.embed_policy == "int8":
        apply_embed_policy(raw, args.embed_policy)
    final = os.path.join(args.workdir, shard_filename(tag))
    if args.no_quant:
        os.replace(raw, final)
    else:
        quant_int4(raw, final)
        try: os.remove(raw)
        except OSError: pass
    print(f"WORKER_OK {tag} {shard_filename(tag)} {os.path.getsize(final)} peakRSS={rss_gb():.2f}GB", flush=True)


# ---------------- dispatch one stage to a host (non-blocking) ----------------
def worker_argv(args, tag, workdir):
    a = ["worker", "--repo", args.repo, "--workdir", workdir, "--tag", tag,
         "--front", str(args.front), "--back", str(args.back), "--interior", str(args.interior),
         "--embed-policy", args.embed_policy, "--seq", str(args.seq)]
    if args.no_quant: a.append("--no-quant")
    return a


def dispatch(args, tag, host):
    log = open(os.path.join(args.workdir, f"_{tag}.log"), "w")
    if host == "local":
        cmd = [VENV_PY, os.path.join(HERE, "osiris_build_distributed.py")] + worker_argv(args, tag, args.workdir)
        env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="0", TRANSFORMERS_VERBOSITY="error")
        return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    conn, rdir, base, _ = parse_ssh(host)
    py = f"{rdir}/.venv/bin/python" if args.remote_venv else "python3"
    rargs = " ".join(worker_argv(args, tag, f"{rdir}/work"))
    rcmd = f"cd {rdir} && HF_HUB_ENABLE_HF_TRANSFER=0 TRANSFORMERS_VERBOSITY=error {py} osiris_build_distributed.py {rargs}"
    return subprocess.Popen(base + [rcmd], stdout=log, stderr=subprocess.STDOUT)


def finalize(args, tag, host):
    if host == "local":
        return True
    conn, rdir, _, scp = parse_ssh(host)
    fn = shard_filename(tag)
    r = subprocess.run(scp + [f"{conn}:{rdir}/work/{fn}", os.path.join(args.workdir, fn)])
    return r.returncode == 0


# ---------------- scheduler ----------------
def run_build(args):
    from transformers import AutoConfig
    from osiris_build_lean import emit_registry, verify_chain
    os.makedirs(args.workdir, exist_ok=True)
    config = AutoConfig.from_pretrained(args.repo)
    nH = config.num_attention_heads
    nKV = getattr(config, "num_key_value_heads", nH)
    hd = getattr(config, "head_dim", config.hidden_size // nH)
    n_layers = config.num_hidden_layers
    F, M, K, cuts, stages = stage_list(n_layers, args.front, args.back, args.interior)
    tags = [s[3] for s in stages]
    print(f"DISTRIBUTED BUILD {args.repo}")
    print(f"plan: FRONT[0:{F}] interior K={K} cuts={cuts} BACK[{M}:{n_layers}]")
    print(f"hosts: {args.hosts}", flush=True)
    print(f"stage->host: " + ", ".join(f"{t}->{args.hosts[i % len(args.hosts)]}" for i, t in enumerate(tags)), flush=True)

    pending = list(enumerate(tags))
    running = {}   # tag -> (proc, host, t0)
    done, failed = [], []
    while pending or running:
        while pending and len(running) < args.max_parallel:
            idx, tag = pending.pop(0)
            host = args.hosts[idx % len(args.hosts)]
            print(f"  [{time.strftime('%H:%M:%S')}] dispatch {tag} -> {host}", flush=True)
            running[tag] = (dispatch(args, tag, host), host, time.time())
        for tag, (p, host, t0) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            running.pop(tag)
            dt = time.time() - t0
            if rc == 0 and finalize(args, tag, host):
                done.append(tag); print(f"  [{time.strftime('%H:%M:%S')}] OK {tag} ({dt:.0f}s on {host})", flush=True)
            else:
                failed.append(tag); print(f"  [{time.strftime('%H:%M:%S')}] FAIL {tag} rc={rc} (see _{tag}.log)", flush=True)
        time.sleep(2)
    if failed:
        sys.exit(f"stages failed: {failed}")

    # assemble: tokenizer + registry (shards ARE the artifacts; no merge)
    for cand in (os.path.join(args.workdir, "_hfsrc", "tokenizer.json"),):
        if os.path.exists(cand):
            shutil.copy(cand, os.path.join(args.workdir, "tokenizer.json")); break
    else:
        # pull tokenizer from the first ssh host if not local
        for h in args.hosts:
            if h != "local":
                conn, rdir, _, scp = parse_ssh(h)
                subprocess.run(scp + [f"{conn}:{rdir}/work/_hfsrc/tokenizer.json",
                                      os.path.join(args.workdir, "tokenizer.json")])
                break
    cfg = dict(id=args.id, name=args.name, nKV=nKV, hd=hd, n_layers=n_layers,
               F=F, M=M, K=K, cuts=cuts, stop=args.stop, default_prompt=args.prompt, def_temp=args.def_temp)
    entry, _ = emit_registry(args.workdir, cfg)
    sizes = {t: os.path.getsize(os.path.join(args.workdir, shard_filename(t))) for t in tags}
    print("\nshards:", {t: f"{s/1048576:.1f}MB" for t, s in sizes.items()}, flush=True)
    print("\n--- registry entry ---\n" + entry, flush=True)
    if not args.no_verify and os.path.exists(os.path.join(args.workdir, "tokenizer.json")):
        # verify_chain reads tokenizer from a repo_cache dir; point it at workdir
        tdir = args.workdir if os.path.exists(os.path.join(args.workdir, "tokenizer.json")) else os.path.join(args.workdir, "_hfsrc")
        os.makedirs(os.path.join(args.workdir, "_vc"), exist_ok=True)
        shutil.copy(os.path.join(args.workdir, "tokenizer.json"), os.path.join(args.workdir, "_vc", "tokenizer.json"))
        dims = (nH, nKV, hd, n_layers, F, M, K, cuts)
        print("\n[verify] chaining the independently-built shards ...", flush=True)
        try:
            txt = verify_chain(args.workdir, os.path.join(args.workdir, "_vc"), dims, args.prompt)
            print("Q:", args.prompt); print("A:", txt)
        except Exception as e:
            print("verify skipped/failed:", e)
    print("\n=== DISTRIBUTED BUILD DONE ===")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="role", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--repo", required=True); w.add_argument("--workdir", required=True); w.add_argument("--tag", required=True)
    w.add_argument("--front", type=int, default=2); w.add_argument("--back", type=int, default=2)
    w.add_argument("--interior", type=int, default=4); w.add_argument("--seq", type=int, default=4)
    w.add_argument("--embed-policy", default="fp16"); w.add_argument("--no-quant", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--repo", required=True); b.add_argument("--workdir", default="dist_out")
    b.add_argument("--front", type=int, default=2); b.add_argument("--back", type=int, default=2)
    b.add_argument("--interior", type=int, default=4); b.add_argument("--seq", type=int, default=4)
    b.add_argument("--embed-policy", default="fp16")
    b.add_argument("--hosts", nargs="+", default=["local"])
    b.add_argument("--max-parallel", type=int, default=2)
    b.add_argument("--remote-venv", action="store_true")
    b.add_argument("--id", default="distmodel"); b.add_argument("--name", default="distributed model")
    b.add_argument("--stop", type=int, default=151645); b.add_argument("--def-temp", type=float, default=0.5)
    b.add_argument("--prompt", default="What is the capital of France?")
    b.add_argument("--no-quant", action="store_true"); b.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    run_worker(args) if args.role == "worker" else run_build(args)


if __name__ == "__main__":
    main()
