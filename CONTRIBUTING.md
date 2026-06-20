# Contributing to Osiris Compute

Thanks for being here. Osiris Compute is a free public utility, and it gets better
the more people poke at it, break it, and send fixes back. Contributions of every
size are welcome, from a typo in the README to a new model architecture in the
partitioner.

## Ways to help

- **Try it and report what happened.** Open the [live grid](https://compute.osirisindustries.net)
  or run it locally, and file an issue with your browser, OS, GPU, and what you saw.
  Real-world device reports are genuinely valuable, especially on phones and on LANs.
- **Improve the docs.** If something in the README or `tools/README.md` was confusing,
  fix it. You are the best judge of what tripped you up.
- **Add or harden a model.** The partitioning toolchain in [`tools/`](tools/) is
  config-driven. New architectures, better seam detection, smaller shards, and
  verification improvements are all high-value.
- **Tackle the networking layer.** WebRTC reconnection, NAT traversal, roster/circle
  UX, and resilience when a peer drops mid-token are open problems.
- **File bugs and ideas.** An issue that clearly describes a problem is a contribution.

## Development setup

```bash
git clone https://github.com/osiris-industries/osiris-compute.git
cd osiris-compute
npm install          # one dependency: ws
node server.js       # http://127.0.0.1:8080
```

To actually run a model, fetch a small demo model's shards (see the README
"Try a real distributed model" and "Run it on a LAN" sections):

```bash
./fetch-demo-model.sh
```

The client needs a **WebGPU-capable browser** (current desktop Chrome or Edge) for
the inference path.

## Pull requests

1. Fork the repo and branch from `main`.
2. Keep changes focused. Small, single-purpose PRs get reviewed and merged faster.
3. The codebase is deliberately tiny and dependency-light. Please keep it that way:
   no new runtime dependencies without a strong reason, and prefer the platform
   (Node builtins, browser APIs) over packages.
4. Test what you touched. If you changed the partitioner, run its CPU verification.
   If you changed the client or server, confirm a circle still forms and a small
   model still streams tokens.
5. Describe what you changed and how you verified it in the PR.

## Licensing of contributions

This project is **AGPL-3.0**. By submitting a contribution you agree to license it
under the same terms. Third-party code or assets you add must be license-compatible
and listed in [NOTICE](NOTICE).

## Conduct

Be decent. See the [Code of Conduct](CODE_OF_CONDUCT.md). Questions, ideas, or
anything else: **admin@osirisindustries.net**.
