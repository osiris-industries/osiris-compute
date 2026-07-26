#!/usr/bin/env python3
"""v0.3 game + notifications:
- Fix score bug (gates/scoring ran after death & while idle) via one shared factory
  that only advances physics when alive & not paused.
- Add pause (button + tap-to-resume).
- Shared circle scoreboard (peers report best -> host aggregates -> broadcasts).
- 'Pipeline ready' toast to host AND peers (vibrates phones) so players know to act.
"""
f = "public/index.html"
s = open(f, encoding="utf-8").read()

def rep(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, f"[{label}] expected 1, found {n}"
    s = s.replace(old, new, 1)
    print("ok", label)

# ---- 1) replace the HOST game IIFE with factory + helpers (index-based, whitespace-proof)
h0 = s.index("// Packet Run — 2D/CPU game, themed to the site palette.")
h1 = s.index("</script>", h0)
FACTORY = r'''// ---- Packet Run (v0.3): one factory for host Play tab + joiners; toast + scoreboard ----
function toast(msg, kind){
  var el=document.getElementById("toast"); if(!el) return;
  var c=(kind==="ok")?"var(--ok)":"var(--rule-strong)";
  el.innerHTML='<div style="background:var(--paper-2);border:1px solid '+c+';color:var(--ink);border-radius:10px;padding:12px 18px;font-size:14px;box-shadow:0 6px 24px rgba(0,0,0,.5);text-align:center">'+String(msg).replace(/[<>]/g,"")+'</div>';
  el.style.display="block";
  clearTimeout(window.__toastT); window.__toastT=setTimeout(function(){ el.style.display="none"; }, 6000);
  try{ if(navigator.vibrate) navigator.vibrate(120); }catch(e){}
}
function pgReport(v){
  if(typeof role!=="undefined" && role==="host" && typeof host!=="undefined" && host.pgSet){ host.pgSet((typeof myName!=="undefined"&&myName)||"host", v); }
  else if(typeof worker!=="undefined" && worker.channel && worker.channel.readyState==="open"){ try{ worker.channel.send(JSON.stringify({cmd:"pg-score", score:v})); }catch(e){} }
}
function pgRender(board){
  var html=(board||[]).map(function(r,i){ return '<div>'+(i+1)+'. '+String(r.name).replace(/[<>]/g,"")+' &mdash; <b>'+(r.score|0)+'</b></div>'; }).join("")||'<div class="muted">no scores yet &mdash; go play</div>';
  var a=document.getElementById("pgBoard"), b=document.getElementById("pgwBoard");
  if(a) a.innerHTML=html; if(b) b.innerHTML=html;
}
function makePacketRun(cv, sEl, bEl, opts){
  opts=opts||{};
  if(!cv) return { setActive:function(){}, pause:function(){} };
  var ctx=cv.getContext("2d"), W=cv.width, H=cv.height, GAP=132;
  var cs=getComputedStyle(document.documentElement), pick=function(n,d){ return (cs.getPropertyValue(n)||"").trim()||d; };
  var PAPER=pick("--paper","#0b0b0c"), OK=pick("--ok","#7fd1a3"), BAD=pick("--bad","#e08a8a"), MUTE=pick("--mute","#8a8a86");
  var best=+(localStorage.getItem("pg_best")||0); if(bEl) bEl.textContent=best;
  var raf=null, y, vy, gates, t, score, dead, active=false, paused=false;
  function reset(){ y=H/2; vy=0; gates=[]; t=0; score=0; dead=false; if(sEl) sEl.textContent=0; }
  function flap(){ if(!active) return; if(paused){ paused=false; return; } if(dead){ reset(); return; } vy=-5.4; }
  function loop(){
    if(!active){ raf=null; return; }
    if(!dead && !paused){
      t++; vy+=0.3; y+=vy;
      if(t%78===0) gates.push({x:W, top:70+Math.random()*(H-140-GAP), done:false});
      for(var i=0;i<gates.length;i++){ var p=gates[i]; p.x-=2.5;
        if(!p.done && p.x+26<70){ p.done=true; score++; if(sEl) sEl.textContent=score; }
        if(85>p.x && 70<p.x+26 && (y-7<p.top || y+7>p.top+GAP)) dead=true;
      }
      gates=gates.filter(function(p){ return p.x>-30; });
      if(y+7>H||y-7<0) dead=true;
      if(dead && score>best){ best=score; localStorage.setItem("pg_best",best); if(bEl) bEl.textContent=best; if(opts.onBest) opts.onBest(best); }
    }
    ctx.fillStyle=PAPER; ctx.fillRect(0,0,W,H);
    ctx.strokeStyle="rgba(255,255,255,.05)"; ctx.lineWidth=1;
    for(var gx=0;gx<W;gx+=40){ ctx.beginPath(); ctx.moveTo(gx,0); ctx.lineTo(gx,H); ctx.stroke(); }
    ctx.fillStyle=OK; ctx.fillRect(70,y-7,15,14);
    for(var k=0;k<gates.length;k++){ var q=gates[k];
      ctx.fillStyle="rgba(255,255,255,.07)"; ctx.strokeStyle="rgba(255,255,255,.16)";
      ctx.fillRect(q.x,0,26,q.top); ctx.strokeRect(q.x+.5,.5,25,q.top);
      ctx.fillRect(q.x,q.top+GAP,26,H-(q.top+GAP)); ctx.strokeRect(q.x+.5,q.top+GAP+.5,25,H-(q.top+GAP)-1);
    }
    ctx.font="15px 'IBM Plex Mono',monospace"; ctx.textAlign="center";
    if(dead){ ctx.fillStyle=BAD; ctx.fillText("tap to retry",W/2,H/2); }
    else if(paused){ ctx.fillStyle=MUTE; ctx.fillText("paused — tap to resume",W/2,H/2); }
    raf=requestAnimationFrame(loop);
  }
  var api={ setActive:function(on){ active=on; if(on){ reset(); paused=false; if(!raf) raf=requestAnimationFrame(loop); } else if(raf){ cancelAnimationFrame(raf); raf=null; } }, pause:function(){ if(active) paused=!paused; } };
  cv.addEventListener("pointerdown", function(e){ e.preventDefault(); if(!active){ api.setActive(true); return; } flap(); });
  document.addEventListener("keydown", function(e){ if(active && (e.code==="Space"||e.code==="ArrowUp")){ e.preventDefault(); flap(); } });
  return api;
}
window.__pg = makePacketRun(document.getElementById("pgCanvas"), document.getElementById("pgScore"), document.getElementById("pgBest"), {onBest:pgReport});
(function(){ var pp=document.getElementById("pgPause"); if(pp) pp.onclick=function(){ window.__pg&&window.__pg.pause(); }; var wp=document.getElementById("pgwPause"); if(wp) wp.onclick=function(){ window.__pgw&&window.__pgw.pause(); }; })();
'''
s = s[:h0] + FACTORY + s[h1:]
print("ok  host game -> factory")

# ---- 2) replace the JOINER game IIFE with a one-line factory call
w0 = s.index("// Packet Run for JOINERS")
w1 = s.index("</script>", w0)
s = s[:w0] + 'window.__pgw = makePacketRun(document.getElementById("pgwCanvas"), document.getElementById("pgwScore"), document.getElementById("pgwBest"), {onBest:pgReport});\n' + s[w1:]
print("ok  joiner game -> factory call")

# ---- 3) toast container after the topbar
rep('<span class="tag">Private compute circle</span>\n  </div>',
    '<span class="tag">Private compute circle</span>\n  </div>\n  <div id="toast" style="position:fixed;left:50%;top:14px;transform:translateX(-50%);z-index:80;display:none;max-width:92vw;pointer-events:none"></div>',
    "toast container")

# ---- 4) host: scoreboard state + methods
rep('    broadcastToPeers(obj){ const sObj=JSON.stringify(obj); for (const p of this.peers.values()){ if (p.channel && p.channel.readyState==="open"){ try{ p.channel.send(sObj); }catch(e){} } } },',
    '    broadcastToPeers(obj){ const sObj=JSON.stringify(obj); for (const p of this.peers.values()){ if (p.channel && p.channel.readyState==="open"){ try{ p.channel.send(sObj); }catch(e){} } } },\n'
    '    pgScores:{},\n'
    '    pgSet(name,score){ if(!(name in this.pgScores)||score>this.pgScores[name]) this.pgScores[name]=score; this.pgBroadcast(); },\n'
    '    pgBroadcast(){ const b=Object.keys(this.pgScores).map(n=>({name:n,score:this.pgScores[n]})).sort((a,b)=>b.score-a.score).slice(0,10); this.broadcastToPeers({cmd:"pg-board",board:b}); pgRender(b); },',
    "host pg methods")

# ---- 5) host onControl: receive peer scores
rep('} else if (m.cmd==="back-out"){ models.onBackOut(m);\n      }',
    '} else if (m.cmd==="back-out"){ models.onBackOut(m);\n      } else if (m.cmd==="pg-score"){ this.pgSet((peer&&peer.name)||("guest-"+peerId.slice(0,4)), m.score|0);\n      }',
    "onControl pg-score")

# ---- 6) worker onTask: ready toast + scoreboard render
rep('if (m.cmd==="roster"){ renderPresence(m.roster); return; }',
    'if (m.cmd==="roster"){ renderPresence(m.roster); return; }\n'
    '      if (m.cmd==="pipeline-ready"){ toast("Circle\'s ready — submit a prompt or watch it run", "ok"); return; }\n'
    '      if (m.cmd==="pg-board"){ pgRender(m.board); return; }',
    "worker onTask handlers")

# ---- 7) host setup: toast + broadcast ready
rep('this.mlog("pipeline ready — "+(M.chat?"ask a question":"type a prompt")+" and hit Generate.");',
    'this.mlog("pipeline ready — "+(M.chat?"ask a question":"type a prompt")+" and hit Generate.");\n'
    '        toast("Pipeline ready — ask a question or start a run", "ok"); if(typeof host!=="undefined" && host.broadcastToPeers) host.broadcastToPeers({cmd:"pipeline-ready"});',
    "setup ready toast+broadcast")

# ---- 8) host Play stage: pause button + scoreboard
rep('<p class="muted" style="margin-top:8px">Tap / click / space to fly the packet through the gates.</p>',
    '<div class="row" style="margin-top:8px"><span class="muted">Tap / click / space to fly.</span><button class="chip" id="pgPause">Pause</button></div>\n'
    '        <div class="muted" style="margin-top:12px">Circle scoreboard</div>\n'
    '        <div class="log" id="pgBoard" style="max-height:120px;margin-top:4px"></div>',
    "host pause+board")

# ---- 9) joiner Play stage: pause button + scoreboard
rep('<p class="muted" style="margin-top:8px">Tap / click / space to fly the packet.</p>',
    '<div class="row" style="margin-top:8px"><span class="muted">Tap / click / space to fly.</span><button class="chip" id="pgwPause">Pause</button></div>\n'
    '        <div class="muted" style="margin-top:12px">Circle scoreboard</div>\n'
    '        <div class="log" id="pgwBoard" style="max-height:120px;margin-top:4px"></div>',
    "joiner pause+board")

open(f, "w", encoding="utf-8").write(s)
print("wrote", f, len(s), "chars")
