#!/usr/bin/env python3
"""v0.3 UI: navbar tabs inside the circle screen + themed 'Play' game.
- Removes the off-brand floating FAB/overlay (which wrongly showed on the landing menu).
- Adds a tab strip inside #hostView (so it can't appear on the front menu).
- Restyles the game to the site theme (paper/ink/mono; no blue).
Idempotent-guarded: each replace asserts exactly one match.
"""
f = "public/index.html"
s = open(f, encoding="utf-8").read()

def rep(old, new, label):
    global s
    n = s.count(old)
    assert n == 1, f"[{label}] expected 1, found {n}"
    s = s.replace(old, new, 1)
    print("ok", label)

# 1) remove the injected floating play block (comment -> just before kernel-src)
start = s.index("<!-- ===== Play-while-you-wait")
anchor = '<script id="kernel-src" type="text/plain">'
end = s.index(anchor)
s = s[:start] + s[end:]
print("ok  removed floating FAB/overlay block")

# 2) tab CSS before </head>
rep("</head>",
"""<style id="osiris-tabs">
  #circleTabs{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 22px;padding-bottom:16px;border-bottom:1px solid var(--rule)}
  #circleTabs .chip{font-size:12px;padding:7px 15px}
  .tab-hidden{display:none!important}
</style>
</head>""", "tab css")

# 3a) h1 + nav + circle stage id/class
rep(
'''      <h1>Your circle is <em>live</em>.</h1>
      <div class="stage">
        <div class="row">''',
'''      <h1>Your circle is <em>live</em>.</h1>
      <nav id="circleTabs">
        <button class="chip active" data-tab="circle">Circle</button>
        <button class="chip" data-tab="model">Model</button>
        <button class="chip" data-tab="play">Play</button>
        <button class="chip" data-tab="live">Live</button>
        <button class="chip" data-tab="lab">Lab</button>
        <button class="chip" data-tab="output">Output</button>
      </nav>
      <div class="stage tab-panel" id="circleStage" data-tab="circle">
        <div class="row">''', "nav + circle stage")

# 3b-e) tag the other stages as tab panels
rep('<div class="stage" id="outStage">',   '<div class="stage tab-panel" id="outStage" data-tab="output">',  "outStage")
rep('<div class="stage" id="sceneStage">', '<div class="stage tab-panel" id="sceneStage" data-tab="live">',   "sceneStage")
rep('<div class="stage" id="modelStage">', '<div class="stage tab-panel" id="modelStage" data-tab="model">',  "modelStage")

# 3f) insert Play stage right before labStage, and tag labStage
rep('<div class="stage" id="labStage">',
'''<div class="stage tab-panel" id="playStage" data-tab="play">
        <div class="row" style="margin-bottom:6px">
          <h2 style="font-family:var(--serif);font-weight:400;font-size:22px">Play while you wait</h2>
          <span class="pill">score&nbsp;<span id="pgScore">0</span> &middot; best&nbsp;<span id="pgBest">0</span></span>
        </div>
        <p class="muted">Splitting a model across devices takes a little longer than a cloud service &mdash; so pass the time. Runs on your CPU, never the GPU the circle needs.</p>
        <canvas id="pgCanvas" width="440" height="480" style="max-width:440px;touch-action:none"></canvas>
        <p class="muted" style="margin-top:8px">Tap / click / space to fly the packet through the gates.</p>
      </div>

      <div class="stage tab-panel" id="labStage" data-tab="lab">''', "playStage + labStage")

# 4) tab controller + themed game, injected before kernel-src
inject = r'''<script>
// v0.3 circle tabs: one stage visible at a time, inside the circle screen only.
(function(){
  var nav=document.getElementById("circleTabs"); if(!nav) return;
  var panels=[].slice.call(document.querySelectorAll("#hostView .tab-panel"));
  function activate(name){
    panels.forEach(function(p){ p.classList.toggle("tab-hidden", p.getAttribute("data-tab")!==name); });
    [].slice.call(nav.children).forEach(function(b){ b.classList.toggle("active", b.getAttribute("data-tab")===name); });
    if(window.__pg) window.__pg.setActive(name==="play");
  }
  nav.addEventListener("click", function(e){ var b=e.target.closest("[data-tab]"); if(b) activate(b.getAttribute("data-tab")); });
  activate("circle");
})();
// Packet Run — 2D/CPU game, themed to the site palette. Runs only while the Play tab is active.
(function(){
  var cv=document.getElementById("pgCanvas"); if(!cv) return;
  var ctx=cv.getContext("2d"), W=cv.width, H=cv.height, GAP=132;
  var cs=getComputedStyle(document.documentElement), pick=function(n,d){ return (cs.getPropertyValue(n)||"").trim()||d; };
  var INK=pick("--ink","#f1f1ef"), PAPER=pick("--paper","#0b0b0c"), OK=pick("--ok","#7fd1a3"), BAD=pick("--bad","#e08a8a");
  var sEl=document.getElementById("pgScore"), bEl=document.getElementById("pgBest");
  var best=+(localStorage.getItem("pg_best")||0); if(bEl) bEl.textContent=best;
  var raf=null, y, vy, gates, t, score, dead, active=false;
  function reset(){ y=H/2; vy=0; gates=[]; t=0; score=0; dead=false; if(sEl) sEl.textContent=0; }
  function flap(){ if(!active) return; if(dead){ reset(); return; } vy=-5.4; }
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
  window.__pg={ setActive:function(on){ active=on; if(on){ reset(); if(!raf) raf=requestAnimationFrame(loop); } else if(raf){ cancelAnimationFrame(raf); raf=null; } } };
  cv.addEventListener("pointerdown", function(e){ e.preventDefault(); flap(); });
  document.addEventListener("keydown", function(e){ if(active && (e.code==="Space"||e.code==="ArrowUp")){ e.preventDefault(); flap(); } });
})();
</script>
'''
rep(anchor, inject + anchor, "tab+game JS")

open(f, "w", encoding="utf-8").write(s)
print("wrote", f, len(s), "chars")
