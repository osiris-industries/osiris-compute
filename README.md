# Osiris Compute

**A browser-based, peer-to-peer public compute utility. No install. No cryptocurrency. No data center.**

Osiris Compute lets people pool the spare compute of their own devices and trusted peers into a **private circle** — straight from the browser. It is a modern, web-era successor to volunteer-computing projects like BOINC and SETI@home, built for one stubborn belief: **useful computation should not have to live inside a hyperscaler's data center.**

Live grid: **https://compute.osirisindustries.net**

---

## Why this exists

Compute has quietly re-centralized. Training and inference, the work that increasingly matters, happens on a handful of corporate clouds behind capacity gates and credit cards. Meanwhile billions of capable devices — laptops, desktops, phones — sit idle most of the day.

Osiris Compute is a small bet that you can put that idle capacity to work **without** an account, an installer, a token, or a wallet — and without a server ever seeing your data. You open a tab, you share a link with people you trust, and the browsers do the rest.

It is deliberately a **free public utility, not a product.** Fork it, host it, federate it.

---

## How it works

1. **Open a circle.** One device hosts and gets a single share link.
2. **Peers join from the browser.** No install — just the link.
3. **The coordination server only introduces, then withdraws.** It performs WebRTC signaling (SDP/ICE exchange) and roster/capability discovery, then steps out of the data path.
4. **Work flows directly peer-to-peer** over WebRTC data channels. Tasks execute inside a **WebAssembly + WebGPU sandbox**, isolated from files, photos, passwords, and the local network.
5. **Participation is conscious.** Computation runs only while the tab is open and visible — no hidden background work, no surprise battery drain.

The server never sees your data, because after the introduction it isn't in the conversation.

### Distributed LLM inference (the headline capability)

Osiris Compute can run a language model **too big for any single device** by splitting its layers across the circle (pipeline parallelism):

- The heavy **embedding + head** run on capable "anchor" devices (a laptop/desktop).
- The **interior transformer layers** are sliced into shards that phones can each hold.
- Each generated token, only a few **kilobytes of hidden-state** travel device → device — not the weights.
- Placement is capability-aware: strong devices anchor, weak devices hold a slice.

The live grid currently serves five models this way — from Qwen2.5-0.5B up to **Mistral-7B-Instruct split across a host + 6 interior shards**, with coherent generation across machines that could never each hold the whole model.

---

## Quickstart

```bash
npm install            # ws (see package.json); server uses node's built-in http
node server.js         # signaling router + static host on $PORT (default 8080)
# open http://localhost:8080 in two browser tabs / two devices
```

### Configuration (environment variables)

| var | purpose | default |
|---|---|---|
| `PORT` / `HOST` | where the signaling + static server listens | `8080` / `0.0.0.0` |
| `TURN_SECRET` | shared secret for time-limited TURN credentials (HMAC-SHA1) | — (TURN off if unset) |
| `TURN_HOST` / `TURN_PORT` | your coturn relay, for peers behind symmetric NATs | — |

STUN uses public `stun.l.google.com:19302` out of the box. TURN is **optional** — only needed so peers on hostile NATs can still connect; supply your own relay. No secrets are committed to this repo.

---

## Repository layout

```
server.js          signaling router (WebRTC SDP/ICE relay, roster, ICE-config endpoint) + static host
public/index.html  the client — WebRTC mesh, WASM/WebGPU sandbox, model registry, shard chaining
public/wabt.js     vendored WebAssembly Binary Toolkit (Apache-2.0; see NOTICE)
```

Model **shards** (the `.onnx` files) are served as static assets out-of-band (not in this repo). The partitioning toolchain that produces browser-ready FRONT / interior / BACK shards from a HuggingFace model is a separate piece of the project.

---

## License

**GNU Affero General Public License v3.0** (see [LICENSE](LICENSE)). If you run a modified version as a network service, the AGPL asks you to share your changes — fitting for a utility meant to stay public. Third-party components are listed in [NOTICE](NOTICE).

A project of **Osiris Industries** — admin@osirisindustries.net
