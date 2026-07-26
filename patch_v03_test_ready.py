#!/usr/bin/env python3
"""v0.3: make the client turnkey for the 6-phone test.
- Default model = qwen15p (phone-sharded); default 'head on its own device' = ON.
- Hide model picker / solo / offload behind an 'Advanced' toggle in the Model tab.
- Hide the Scene director block and the Live tab (not needed for the LLM test).
- Give joiners (worker view) the Packet Run game too.
"""
f = "public/index.html"
s = open(f, encoding="utf-8").read()

def rep(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, f"[{label}] expected 1, found {n}"
    s = s.replace(old, new, 1)
    print("ok", label)

# --- defaults ---
rep('<option value="qwen15p">', '<option value="qwen15p" selected>', "default model qwen15p")
rep('<input type="checkbox" id="offloadBack">', '<input type="checkbox" id="offloadBack" checked>', "offload default ON")

# --- advanced disclosure: tag the knobs + add a toggle ---
rep('<select id="mModel" style="background:#0e1217', '<select id="mModel" class="adv-item" style="background:#0e1217', "mModel adv-item")
rep('<button class="btn" id="soloBtn">Run on this device (no circle)</button>',
    '<button class="btn ghost adv-item" id="soloBtn">Run on this device (no circle)</button>', "solo adv-item")
rep('<label id="offloadWrap" style="display:inline-flex', '<label id="offloadWrap" class="adv-item" style="display:inline-flex', "offload adv-item")
rep('<button class="btn" id="shardBtn">Set up the model pipeline</button>',
    '<button class="btn" id="shardBtn">Set up the model pipeline</button>\n            <button class="chip" id="advToggle" title="Model &amp; placement options">⚙ Advanced</button>', "advToggle button")

# --- advanced CSS ---
rep('.tab-hidden{display:none!important}',
'''.tab-hidden{display:none!important}
  .adv-item{display:none!important}
  #modelStage.adv-open select.adv-item{display:inline-block!important}
  #modelStage.adv-open button.adv-item{display:inline-block!important}
  #modelStage.adv-open label.adv-item{display:inline-flex!important}''', "adv css")

# --- hide scene director block ---
rep('<div style="margin-top:14px;border-top:1px dashed var(--rule);padding-top:12px">\n          <div class="row">\n            <div>\n              <h2 style="font-family:var(--serif);font-weight:400;font-size:18px">Scene director',
    '<div class="hidden" style="margin-top:14px;border-top:1px dashed var(--rule);padding-top:12px">\n          <div class="row">\n            <div>\n              <h2 style="font-family:var(--serif);font-weight:400;font-size:18px">Scene director', "hide scene director")

# --- hide Live tab ---
rep('<button class="chip" data-tab="live">Live</button>', '<button class="chip hidden" data-tab="live">Live</button>', "hide Live tab")

# --- worker Play stage (joiners get the game) ---
rep('      <div class="stage hidden" id="wOutStage">',
'''      <div class="stage" id="wPlayStage">
        <div class="row" style="margin-bottom:6px">
          <h2 style="font-family:var(--serif);font-weight:400;font-size:20px">Play while you wait</h2>
          <span class="pill">score&nbsp;<span id="pgwScore">0</span> &middot; best&nbsp;<span id="pgwBest">0</span></span>
        </div>
        <p class="muted">Your device is downloading its shard and helping the circle think &mdash; pass the time. Runs on your CPU, never the GPU the circle needs.</p>
        <canvas id="pgwCanvas" width="440" height="480" style="max-width:440px;touch-action:none"></canvas>
        <p class="muted" style="margin-top:8px">Tap / click / space to fly the packet.</p>
      </div>

      <div class="stage hidden" id="wOutStage">''', "worker play stage")

# --- worker game IIFE + advToggle handler ---
inject = r'''<script>
// Advanced toggle in the Model tab (reveals model picker / solo / offload).
(function(){ var b=document.getElementById("advToggle"); if(!b) return;
  b.addEventListener("click", function(){ var ms=document.getElementById("modelStage"); if(ms){ ms.classList.toggle("adv-open"); b.classList.toggle("active"); } }); })();
// Packet Run for JOINERS — 2D/CPU, own canvas. Starts on join (see onJoined) or first tap.
(function(){
  var cv=document.getElementById("pgwCanvas"); if(!cv) return;
  var ctx=cv.getContext("2d"), W=cv.width, H=cv.height, GAP=132;
  var cs=getComputedStyle(document.documentElement), pick=function(n,d){ return (cs.getPropertyValue(n)||"").trim()||d; };
  var PAPER=pick("--paper","#0b0b0c"), OK=pick("--ok","#7fd1a3"), BAD=pick("--bad","#e08a8a");
  var sEl=document.getElementById("pgwScore"), bEl=document.getElementById("pgwBest");
  var best=+(localStorage.getItem("pg_best")||0); if(bEl) bEl.textContent=best;
  var raf=null, y, vy, gates, t, score, dead, active=false;
  function reset(){ y=H/2; vy=0; gates=[]; t=0; score=0; dead=false; if(sEl) sEl.textContent=0; }
  function flap(){ if(dead){ reset(); return; } vy=-5.4; }
  function loop(){
    if(!active){ raf=null; return; }
    t++; vy+=0.3; y+=vy;
    if(t%78===0) gates.push({x:W, top:70+Math.random()*(H-140-GAP), done:false});
    ctx.fillStyle=PAPER; ctx.fillRect(0,0,W,H);
    ctx.strokeStyle="rgba(255,255,255,.05)"; ctx.lineWidth=1;
    for(var gx=0;gx<W;gx+=40){ ctx.beginPath(); ctx.moveTo(gx,0); ctx.lineTo(gx,H); ctx.stroke(); }
    ctx.fillStyle=OK; ctx.fillRect(70,y-7,15,14);
    for(var i=0;i<gates.length;i++){ var p=gates[i]; p.x-=2.5;
      ctx.fillStyle="rgba(255,255,255,.07)"; ctx.strokeStyle="rgba(255,255,255,.16)";
      ctx.fillRect(p.x,0,26,p.top); ctx.strokeRect(p.x+.5,.5,25,p.top);
      ctx.fillRect(p.x,p.top+GAP,26,H-(p.top+GAP)); ctx.strokeRect(p.x+.5,p.top+GAP+.5,25,H-(p.top+GAP)-1);
      if(!p.done && p.x+26<70){ p.done=true; score++; if(sEl) sEl.textContent=score; }
      if(85>p.x && 70<p.x+26 && (y-7<p.top || y+7>p.top+GAP)) dead=true;
    }
    gates=gates.filter(function(p){ return p.x>-30; });
    if(y+7>H||y-7<0) dead=true;
    if(dead){ ctx.fillStyle=BAD; ctx.font="15px 'IBM Plex Mono',monospace"; ctx.textAlign="center"; ctx.fillText("tap to retry",W/2,H/2);
      if(score>best){ best=score; localStorage.setItem("pg_best",best); if(bEl) bEl.textContent=best; } }
    raf=requestAnimationFrame(loop);
  }
  window.__pgw={ setActive:function(on){ active=on; if(on){ reset(); if(!raf) raf=requestAnimationFrame(loop); } else if(raf){ cancelAnimationFrame(raf); raf=null; } } };
  cv.addEventListener("pointerdown", function(e){ e.preventDefault(); if(!active){ window.__pgw.setActive(true); } else flap(); });
  document.addEventListener("keydown", function(e){ if(active && (e.code==="Space"||e.code==="ArrowUp")){ e.preventDefault(); flap(); } });
})();
</script>
'''
rep('<script id="kernel-src" type="text/plain">', inject + '<script id="kernel-src" type="text/plain">', "worker game + adv JS")

# --- start joiner game when they land in the circle ---
rep('worker.onJoined(m); }', 'worker.onJoined(m); if(window.__pgw) window.__pgw.setActive(true); }', "onJoined starts game")

open(f, "w", encoding="utf-8").write(s)
print("wrote", f, len(s), "chars")
