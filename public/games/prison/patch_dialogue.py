#!/usr/bin/env python3
"""Turn La Sombra's one-line dialogue into a two-way conversation wired to qwen15p
via the parent compute client (window.parent.postMessage -> osirisGenerate).
Personas per NPC (Spanish-only; bunkmate English), a respect meter, and input guarding."""
f="index.html"; s=open(f,encoding="utf-8").read()
def rep(old,new,label):
    global s; n=s.count(old); assert n==1, f"[{label}] found {n}"; s=s.replace(old,new,1); print("ok",label)

# 1) conversation CSS
rep("</style>",
'''  #dlg{width:min(94vw,560px);max-height:72vh;display:none;flex-direction:column}
  #dlg .hist{flex:1;overflow-y:auto;max-height:40vh;display:flex;flex-direction:column;gap:6px;margin-bottom:8px}
  #dlg .m{font-size:14px;line-height:1.45;padding:6px 10px;border-radius:10px;max-width:85%;word-break:break-word}
  #dlg .m.npc{align-self:flex-start;background:#12212e;border:1px solid #22384a;color:#f1f1ef}
  #dlg .m.you{align-self:flex-end;background:#1b3a2b;border:1px solid #2f5a44;color:#eafff2}
  #dlg .m.sys{align-self:center;color:#8a8a86;font-size:12px;background:none}
  #dlg .inrow{display:flex;gap:8px}
  #dlg #din{flex:1;background:#0b0f14;color:#f1f1ef;border:1px solid #2a333d;border-radius:8px;padding:9px;font:inherit;font-size:14px;outline:none}
  #dlg #dsend{background:#12212e;color:#e8eef2;border:1px solid #2a4356;border-radius:8px;padding:9px 14px;font:inherit}
  #dlg #dsend:disabled{opacity:.5}
</style>''', "dlg css")

# 2) dialogue box markup -> conversation UI
rep('<div id="dlg"><button class="x" id="dx">✕</button><div class="who" id="dwho"></div><div class="line" id="dline"></div></div>',
'''<div id="dlg">
  <button class="x" id="dx">&#10005;</button>
  <div class="who" id="dwho"></div>
  <div style="height:5px;border-radius:3px;background:rgba(255,255,255,.1);margin:4px 0 8px;overflow:hidden"><i id="dmeter" style="display:block;height:100%;width:50%;background:#7fd1a3;transition:width .3s,background .3s"></i></div>
  <div class="hist" id="dhist"></div>
  <div class="inrow"><input id="din" placeholder="type your reply&#8230;" autocomplete="off"><button id="dsend">Send</button></div>
</div>''', "dlg markup")

# 3) npcs with personas + respect
old_npcs='''  var npcs=[
    {cx:1,cy:1,spr:88,name:"Bunkmate",lang:"en",line:"Hey. Keep your head down, cabrón. Don't stare at nobody."},
    {cx:23,cy:5,spr:87,name:"El Jefe",lang:"es",line:"¿Y tú quién eres? No te conozco... cuidado con lo que dices."},
    {cx:31,cy:2,spr:100,name:"Viejo",lang:"es",line:"La comida aquí es una porquería, joven. ¿Tienes cigarros?"},
    {cx:20,cy:5,spr:111,name:"Preso",lang:"es",line:"Oye, gringo. ¿Sabes jugar dominó o no?"},
    {cx:7,cy:9,spr:96,name:"Guardia",lang:"es",line:"¡Camina! No te detengas en el pasillo."},
    {cx:14,cy:14,spr:112,name:"Flaco",lang:"es",line:"Tsss... ¿traes algo para mí? Aquí todo tiene precio."}
  ];'''
new_npcs='''  var npcs=[
    {cx:1,cy:1,spr:88,name:"Miguel (bunkmate)",lang:"en",respect:0,
     line:"Hey. Keep your head down, cabrón. Don't stare at nobody.",
     sys:"You are Miguel, cellmate to a newcomer wrongly jailed in a Mexican prison called La Sombra. You are bilingual and speak to him in ENGLISH. Street-smart, wary, dryly funny, quietly protective of the new guy. Reply in 1-2 SHORT sentences, in character, no narration. At the very end append a hidden tag ||r:N where N is an integer -2..2 = how your respect for him changed this turn. Never break character."},
    {cx:23,cy:5,spr:87,name:"El Jefe",lang:"es",respect:0,
     line:"¿Y tú quién eres? No te conozco... cuidado con lo que dices.",
     sys:"Eres El Jefe, líder de un grupo dentro de una prisión mexicana llamada La Sombra. Hablas SOLO español, nunca inglés. Orgulloso, territorial y peligroso; pones a prueba a los nuevos y no toleras faltas de respeto. Responde en 1-2 frases CORTAS, en personaje, sin narración. Al final añade una etiqueta oculta ||r:N donde N es un entero -2..2 = cuánto cambió tu respeto por él. Nunca rompas el personaje."},
    {cx:31,cy:2,spr:100,name:"El Viejo",lang:"es",respect:0,
     line:"La comida aquí es una porquería, joven. ¿Tienes cigarros?",
     sys:"Eres 'El Viejo', un preso anciano en una prisión mexicana. Hablas SOLO español. Cansado, filosófico, inofensivo; siempre quieres cigarros. 1-2 frases cortas, en personaje. Añade al final ||r:N (-2..2). No rompas el personaje."},
    {cx:20,cy:5,spr:111,name:"Chuy",lang:"es",respect:0,
     line:"Oye, gringo. ¿Sabes jugar dominó o no?",
     sys:"Eres 'Chuy', un preso común en una prisión mexicana. Hablas SOLO español. Casual, chismoso; quieres jugar dominó y hacer amigos o burlarte. 1-2 frases cortas, en personaje. Añade ||r:N (-2..2). No rompas el personaje."},
    {cx:7,cy:9,spr:96,name:"Guardia",lang:"es",respect:0,
     line:"¡Camina! No te detengas en el pasillo.",
     sys:"Eres un guardia de una prisión mexicana. Hablas SOLO español. Autoritario, seco, sin paciencia; das órdenes. 1-2 frases cortas, en personaje. Añade ||r:N (-2..2). No rompas el personaje."},
    {cx:14,cy:14,spr:112,name:"Flaco",lang:"es",respect:0,
     line:"Tsss... ¿traes algo para mí? Aquí todo tiene precio.",
     sys:"Eres 'Flaco', un preso que trafica de todo dentro de la prisión. Hablas SOLO español. Nervioso, calculador; todo tiene precio. 1-2 frases cortas, en personaje. Añade ||r:N (-2..2). No rompas el personaje."}
  ];'''
rep(old_npcs,new_npcs,"npcs personas")

# 4) keydown: don't drive the player while typing in the input
rep('addEventListener("keydown",function(e){ keys[e.key.toLowerCase()]=1; if(e.key===" "||e.key==="e"){e.preventDefault(); tryTalk();} });',
    'addEventListener("keydown",function(e){ if(e.target&&e.target.tagName==="INPUT"){ if(e.key==="Enter") sendReply(); return; } keys[e.key.toLowerCase()]=1; if(e.key===" "||e.key==="e"){e.preventDefault(); tryTalk();} });',
    "keydown guard")

# 5) freeze movement while the dialogue is open
rep('var sp=1.5, nx=player.x, ny=player.y;',
    'var sp=(document.getElementById("dlg").style.display==="flex")?0:1.5, nx=player.x, ny=player.y;',
    "freeze while talking")

# 6) replace the stub tryTalk + dx handler with the conversation system
rep('  document.getElementById("dx").addEventListener("click",function(){document.getElementById("dlg").style.display="none";});',
    '''  document.getElementById("dx").addEventListener("click",function(){ document.getElementById("dlg").style.display="none"; CONV.npc=null; });
  document.getElementById("dsend").addEventListener("click",sendReply);''', "dsend wire")

old_talk='''  function tryTalk(){ var n=nearNpc(); if(!n) return; var d=document.getElementById("dlg");
    document.getElementById("dwho").textContent=n.name+(n.lang==="es"?"  ·  español":"  ·  english");
    document.getElementById("dline").textContent=n.line; d.style.display="block"; }'''
new_talk='''  var CONV={npc:null, busy:false};
  function esc(t){ return String(t).replace(/[<>]/g,function(c){return c==="<"?"&lt;":"&gt;";}); }
  function addMsg(cls,text){ var h=document.getElementById("dhist"), d=document.createElement("div"); d.className="m "+cls; d.innerHTML=esc(text); h.appendChild(d); h.scrollTop=h.scrollHeight; return d; }
  function setMeter(n){ var f=document.getElementById("dmeter"), p=Math.max(0,Math.min(100,50+n.respect*9)); f.style.width=p+"%"; f.style.background=n.respect<=-4?"#e08a8a":(n.respect<0?"#e6c07b":"#7fd1a3"); }
  function parseMood(t){ var m=t.match(/\\|\\|\\s*r\\s*:\\s*(-?\\d)/i); var mood=m?parseInt(m[1],10):0; return {text:t.replace(/\\s*\\|\\|\\s*r\\s*:\\s*-?\\d.*$/i,"").trim(), mood:Math.max(-2,Math.min(2,mood))}; }
  function tryTalk(){ var n=nearNpc(); if(!n) return; CONV.npc=n;
    document.getElementById("dwho").textContent=n.name+"  ·  "+(n.lang==="en"?"english":"español");
    document.getElementById("dhist").innerHTML=""; addMsg("npc",n.line); setMeter(n);
    document.getElementById("dlg").style.display="flex"; setTimeout(function(){ var i=document.getElementById("din"); if(i) i.focus(); },60); }
  function askBrain(npc,userText){ return new Promise(function(resolve){
    var id="npc_"+Math.random().toString(36).slice(2), done=false;
    function fin(r){ if(done)return; done=true; clearTimeout(to); window.removeEventListener("message",onMsg); resolve(r); }
    function onMsg(e){ var m=e.data; if(!m||m.type!=="osiris-npc-reply"||m.id!==id) return; fin(m.error?{err:m.error}:{text:m.text||""}); }
    var to=setTimeout(function(){ fin({err:"timeout"}); },120000);
    window.addEventListener("message",onMsg);
    try{ window.parent.postMessage({type:"osiris-npc",id:id,sys:npc.sys,userText:userText,max:70,temp:0.7},"*"); }catch(x){ fin({err:"no host"}); } }); }
  function sendReply(){ if(CONV.busy||!CONV.npc) return; var inp=document.getElementById("din"); var text=(inp.value||"").trim(); if(!text) return;
    inp.value=""; addMsg("you",text); CONV.busy=true; document.getElementById("dsend").disabled=true; var typing=addMsg("sys","\\u2026 (the circle is thinking)");
    askBrain(CONV.npc,text).then(function(r){ typing.remove();
      if(r.err){ addMsg("sys", (r.err==="model not ready"||r.err==="no host") ? "(set up the model in the Model tab first \\u2014 NPCs think on the circle's brain)" : "(no answer \\u2014 "+r.err+")"); }
      else { var p=parseMood(r.text); addMsg("npc", p.text||"\\u2026"); CONV.npc.respect=Math.max(-6,Math.min(6,CONV.npc.respect+p.mood)); setMeter(CONV.npc);
             if(CONV.npc.respect<=-5) addMsg("sys","\\u26a0 te has ganado un enemigo\\u2026"); }
      CONV.busy=false; document.getElementById("dsend").disabled=false; }); }'''
rep(old_talk,new_talk,"conversation system")

open(f,"w",encoding="utf-8").write(s); print("wrote",f,len(s),"chars")
