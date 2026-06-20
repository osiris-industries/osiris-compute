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

- **Compute runs in a WASM + WebGPU sandbox** in the browser, isolated from the host
  filesystem, local network, and other origins by the browser's own boundaries.
- **The coordination server only does WebRTC signaling** (SDP/ICE exchange) and roster
  discovery, then steps out of the data path. It is not in the peer-to-peer data flow
  and does not see computation payloads.
- **Circles are trust-scoped.** You share a link with people you choose; the model is
  "compute among peers you trust," not "open compute for anonymous strangers." Treat a
  circle link like any other shared secret.
- **TURN credentials are time-limited** (HMAC-SHA1 over an expiry) and are never
  committed to the repo; they are minted by the server from an environment secret.
- **No secrets in the repository.** Relay credentials, tokens, and host config are
  injected via environment variables at deploy time.

## Scope

In scope: the client (`public/`), the signaling server (`server.js`), and the
partitioning toolchain (`tools/`). Out of scope: third-party dependencies (report
those upstream) and the security of any fork's own deployment/configuration.
