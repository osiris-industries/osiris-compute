# Changelog

All notable changes to Osiris Compute are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/). Versioning: [SemVer](https://semver.org/).

## [0.2.0] - 2026-07-23
Reconciliation release: the repository is brought back in sync with the live
client that had been hardened directly on the router. Everything below shipped
between 0.1.0 and now.

### Added
- **Phone-sharded model `qwen15p`** — Qwen2.5-1.5B in 6 shards, every shard < 300 MB
  (int8 embedding on FRONT, int4 head on BACK).
- **BACK-anchor offload** — opt-in "head on its own device" toggle. Host keeps FRONT
  (embedder); the output head runs on a peer that samples locally and returns only a
  token id, so per-hop wire cost equals a normal mid hop. Enables one-shard-per-device.
- **WASM execution-provider fallback** in session creation, so a device without WebGPU
  runs on CPU instead of hanging on the pipeline-ready timeout.

### Changed
- Router logs no longer record client IPs (privacy).

### Fixed
- Offload engaged only with >= 2 peers, silently keeping BACK on the host; now engages
  with >= 1 peer (a lone peer holds mids + head).
- `setup()` did not reset solo-mode state, so a prior solo run could crash generation
  (`this.soloMids`); solo state is now cleared on every circle build.
- Rendezvous: stale-room join retries are capped and the dead room cleared, ending the
  zombie-retry loop against closed circles.
- "answered across N devices" counted distinct machines instead of pipeline roles.

### Proven
- First successful distributed generation with the head offloaded across two physical
  devices (2026-07-20): iPhone host (FRONT only) + laptop (mids + head), coherent output.

## [0.1.0]
- Initial tagged release: WebRTC signaling router, browser client, pipeline sharding
  for the Qwen2.5 family (0.5B / 1.5B / 3B), Qwen3-32B and Qwen3-4B shard chains,
  general-purpose WGSL/WASM compute lab.
