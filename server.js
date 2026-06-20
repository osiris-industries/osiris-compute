'use strict';
/*
 * Osiris Compute — Signaling Router (v2: chain topology)
 * ------------------------------------------------------
 * A zero-trust WebRTC signaling server for private compute circles.
 *
 * v1 was host-centric (star): the server introduced a host and its peers and
 * relayed opaque SDP/ICE between host<->peer. v2 adds DISCOVERY so any member
 * can connect directly to any other member — enabling a pipeline chain
 * (strong-node -> phone -> strong-node) for distributed model inference.
 *
 * Two additive, backward-compatible things:
 *   1. Members may advertise opaque `caps` (memory budget, gpu, label). The
 *      server stores and echoes them but never interprets them.
 *   2. On any membership change the server broadcasts a `roster` to every member
 *      so peers learn each other's ids+caps and can signal directly. The
 *      existing star flow (created/joined/peer-joined/signal/report) is
 *      unchanged; old clients simply ignore the new `roster` message.
 *
 * The relay still only forwards handshake blobs — never a byte of computation.
 * No database, no monetization. In-memory rooms only.
 * Licensed under AGPL-3.0-or-later. (c) 2026 Osiris Industries.
 */

const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { WebSocketServer } = require('ws');

const PORT = process.env.PORT || 8080;
const HOST = process.env.HOST || '127.0.0.1';
const PUBLIC_DIR = path.join(__dirname, 'public');

const TURN_SECRET = process.env.TURN_SECRET || '';
const TURN_HOST = process.env.TURN_HOST || '';
const TURN_PORT = process.env.TURN_PORT || '3478';
const TURN_TTL = parseInt(process.env.TURN_TTL || '3600', 10);

const rooms = new Map(); // code -> { host, peers: Map<id,ws>, created }

function logE(...args) { console.log(new Date().toISOString(), ...args); }
const sid = (id) => (id ? String(id).slice(0, 6) : '------');

function iceServers() {
  const servers = [{ urls: 'stun:stun.l.google.com:19302' }];
  if (TURN_SECRET && TURN_HOST) {
    const username = (Math.floor(Date.now() / 1000) + TURN_TTL) + ':osiris';
    const credential = crypto.createHmac('sha1', TURN_SECRET).update(username).digest('base64');
    servers.push({ urls: `stun:${TURN_HOST}:${TURN_PORT}` });
    servers.push({
      urls: [`turn:${TURN_HOST}:${TURN_PORT}?transport=udp`, `turn:${TURN_HOST}:${TURN_PORT}?transport=tcp`],
      username, credential,
    });
  }
  return servers;
}

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.svg': 'image/svg+xml', '.png': 'image/png',
  '.ico': 'image/x-icon', '.json': 'application/json', '.webmanifest': 'application/manifest+json',
  '.onnx': 'application/octet-stream', '.wasm': 'application/wasm',
};

const server = http.createServer((req, res) => {
  const urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
  if (urlPath === '/healthz') { res.writeHead(200, { 'content-type': 'text/plain' }); return res.end('ok'); }
  if (urlPath === '/ice') {
    res.writeHead(200, { 'content-type': 'application/json', 'cache-control': 'no-store' });
    return res.end(JSON.stringify({ iceServers: iceServers(), turn: !!(TURN_SECRET && TURN_HOST) }));
  }
  if (urlPath === '/stats') {
    const out = { now: new Date().toISOString(), connections: wss.clients.size, roomCount: rooms.size, rooms: [] };
    for (const [code, room] of rooms) {
      out.rooms.push({
        code, host: sid(room.host && room.host.peerId), peerCount: room.peers.size,
        peers: [...room.peers.keys()].map(sid), ageSec: Math.round((Date.now() - room.created) / 1000),
        lastReport: room.lastReport || null,
      });
    }
    res.writeHead(200, { 'content-type': 'application/json' });
    return res.end(JSON.stringify(out, null, 2));
  }
  let rel = path.normalize(urlPath).replace(/^(\.\.[\/\\])+/, '');
  if (rel === '/' || rel === '\\' || rel === '') rel = '/index.html';
  const filePath = path.join(PUBLIC_DIR, rel);
  if (!filePath.startsWith(PUBLIC_DIR)) { res.writeHead(403); return res.end('forbidden'); }
  fs.readFile(filePath, (err, data) => {
    if (err) {
      return fs.readFile(path.join(PUBLIC_DIR, 'index.html'), (e2, shell) => {
        if (e2) { res.writeHead(404); return res.end('not found'); }
        res.writeHead(200, { 'content-type': TYPES['.html'] }); res.end(shell);
      });
    }
    const ext = path.extname(filePath);
    res.writeHead(200, { 'content-type': TYPES[ext] || 'application/octet-stream' });
    res.end(data);
  });
});

const wss = new WebSocketServer({ server, path: '/ws', maxPayload: 256 * 1024 });
function send(ws, obj) { if (ws && ws.readyState === ws.OPEN) ws.send(JSON.stringify(obj)); }
function randId(n) { return crypto.randomBytes(n).toString('hex'); }

function newRoomCode() {
  const alphabet = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';
  let code = '';
  do { const b = crypto.randomBytes(6); code = ''; for (let i = 0; i < 6; i++) code += alphabet[b[i] % alphabet.length]; }
  while (rooms.has(code));
  return code;
}

// --- v2: roster discovery ---------------------------------------------------
function roomMembers(room) {
  const m = [];
  if (room.host) m.push({ id: room.host.peerId, role: 'host', caps: room.host.caps || null });
  for (const ws of room.peers.values()) m.push({ id: ws.peerId, role: 'peer', caps: ws.caps || null });
  return m;
}
function broadcastRoster(room) {
  if (!room) return;
  const members = roomMembers(room);
  const msg = { type: 'roster', members };
  if (room.host) send(room.host, msg);
  for (const ws of room.peers.values()) send(ws, msg);
}

wss.on('connection', (ws, req) => {
  ws.peerId = randId(8);
  ws.roomId = null; ws.role = null; ws.isAlive = true; ws.caps = null;
  ws.on('pong', () => { ws.isAlive = true; });
  logE('conn-open', sid(ws.peerId), 'from', (req.headers['x-real-ip'] || req.socket.remoteAddress || '?'));

  ws.on('message', (raw) => {
    let msg; try { msg = JSON.parse(raw); } catch { return; }
    if (!msg || typeof msg.type !== 'string') return;

    switch (msg.type) {
      case 'create': {
        if (msg.caps && typeof msg.caps === 'object') ws.caps = msg.caps;
        const code = newRoomCode();
        rooms.set(code, { host: ws, peers: new Map(), created: Date.now() });
        ws.roomId = code; ws.role = 'host';
        send(ws, { type: 'created', roomId: code, selfId: ws.peerId });
        broadcastRoster(rooms.get(code));
        logE('create', code, 'host', sid(ws.peerId));
        break;
      }
      case 'join': {
        const code = String(msg.roomId || '').toUpperCase().trim();
        const room = rooms.get(code);
        if (!room) { send(ws, { type: 'error', error: 'no-such-room' }); logE('join', code, 'peer', sid(ws.peerId), 'REJECTED'); break; }
        if (msg.caps && typeof msg.caps === 'object') ws.caps = msg.caps;
        ws.roomId = code; ws.role = 'peer';
        room.peers.set(ws.peerId, ws);
        send(ws, { type: 'joined', roomId: code, selfId: ws.peerId, hostId: room.host.peerId });
        send(room.host, { type: 'peer-joined', peerId: ws.peerId, caps: ws.caps || null });
        broadcastRoster(room);
        logE('join', code, 'peer', sid(ws.peerId), '| peers now', room.peers.size);
        break;
      }
      // v2: a member updates its capabilities after async probing (e.g. WebGPU limits)
      case 'caps': {
        const room = rooms.get(ws.roomId);
        if (!room) break;
        if (msg.caps && typeof msg.caps === 'object') ws.caps = msg.caps;
        broadcastRoster(room);
        break;
      }
      case 'signal': {
        const room = rooms.get(ws.roomId);
        if (!room) { logE('signal', 'DROP no-room from', sid(ws.peerId)); break; }
        let target = null;
        if (room.host && room.host.peerId === msg.to) target = room.host;
        else if (room.peers) target = room.peers.get(msg.to);
        const kind = msg.data && msg.data.sdp ? ('sdp:' + (msg.data.sdp.type || '?'))
                   : (msg.data && msg.data.candidate ? 'ice' : '?');
        if (target) { send(target, { type: 'signal', from: ws.peerId, data: msg.data }); logE('relay', ws.roomId, sid(ws.peerId), '->', sid(msg.to), kind); }
        else logE('relay', ws.roomId, 'DROP target-gone', sid(ws.peerId), '->', sid(msg.to), kind);
        break;
      }
      case 'report': {
        const room = rooms.get(ws.roomId);
        if (!room || ws.role !== 'host') break;
        room.lastReport = { at: new Date().toISOString(), elapsedSec: msg.elapsedSec, devices: msg.devices };
        logE('report', ws.roomId, 'elapsed', (msg.elapsedSec + 's'), JSON.stringify(msg.devices));
        break;
      }
    }
  });

  ws.on('close', () => {
    const room = rooms.get(ws.roomId);
    logE('conn-close', ws.role || 'none', sid(ws.peerId), 'room', ws.roomId || '-');
    if (!room) return;
    if (ws.role === 'host') {
      for (const peer of room.peers.values()) send(peer, { type: 'host-left' });
      rooms.delete(ws.roomId);
      logE('room-closed', ws.roomId, '(host left)');
    } else {
      room.peers.delete(ws.peerId);
      send(room.host, { type: 'peer-left', peerId: ws.peerId });
      broadcastRoster(room);
    }
  });
});

const heartbeat = setInterval(() => {
  wss.clients.forEach((ws) => {
    if (!ws.isAlive) return ws.terminate();
    ws.isAlive = false;
    try { ws.ping(); } catch { /* noop */ }
  });
}, 30000);
wss.on('close', () => clearInterval(heartbeat));

server.listen(PORT, HOST, () => {
  logE(`[osiris-compute] signaling router v2 on http://${HOST}:${PORT}  (ws:/ws, roster on, turn:${TURN_HOST ? 'on' : 'off'})`);
});
