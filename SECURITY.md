# Security Policy

Osiris Compute runs untrusted-peer computation in the browser, so we take the
threat model seriously and welcome reports.

## Reporting a vulnerability

Please report security issues privately to **admin@osirisindustries.net** rather than
opening a public issue. Include:

- a description of the issue and its impact,
- steps to reproduce (a proof of concept if you have one),
- the affected component (client, signaling server, or partitioning toolchain),
- your environment (browser, OS, version).

We aim to acknowledge reports within a few days and will keep you updated on a fix.
Coordinated disclosure is appreciated: give us a reasonable window to ship a patch
before going public.

## Design notes (the threat model in brief)

- **LLM inference is data-only**: model shards run through onnxruntime-web (WebGPU/WASM);
  no peer-supplied code executes for the inference path.
- **The general-compute feature runs peer-supplied code** (JS in a Web Worker, or WASM/WGSL).
  The Worker has no filesystem or page/DOM access, but it **can make network requests** — so it
  is NOT a full capability sandbox. A peer's module only runs **after the receiving user
  explicitly approves it** (consent prompt, default deny; approval is **per circle** — one OK covers that circle's session, not each individual module). Treat a circle as "compute among
  people you trust," not "run anonymous strangers' code safely."
- Browser-origin boundaries (no host filesystem access, same-origin policy) still apply to all paths.
- **The coordination server only does WebRTC signaling** (SDP/ICE exchange) and roster
  discovery, then steps out of the data path. It is not in the peer-to-peer data flow
  and does not see computation payloads.
- **Circles are trust-scoped.** You share a link with people you choose; the model is
  "compute among peers you trust," not "open compute for anonymous strangers." Treat a
  circle link like any other shared secret.
- **Room codes are the only access secret (zero-trust by design).** There is no per-host
  ownership token: automatic host-reconnect reclaims a code only while it is free, but a
  consequence is that if a host drops, anyone who *already knows* the code could recreate it
  during the gap and adopt returning peers. Same mitigation as any shared secret — don't post
  circle links publicly, and start a fresh circle if a link leaks.
- **TURN credentials are time-limited** (HMAC-SHA1 over an expiry) and are never
  committed to the repo; they are minted by the server from an environment secret.
- **No secrets in the repository.** Relay credentials, tokens, and host config are
  injected via environment variables at deploy time.

## Scope

In scope: the client (`public/`), the signaling server (`server.js`), and the
partitioning toolchain (`tools/`). Out of scope: third-party dependencies (report
those upstream) and the security of any fork's own deployment/configuration.
