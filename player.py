"""
2-D sampling player with highlighted MISER decisions.

The Python side turns the recorded MISER events (pilot draws, split
decisions, leaf draws, in evaluation order) and the plain Monte Carlo
samples into one JSON payload. The browser replays both runs side by side
on canvas layers. At each split decision up to a chosen depth the replay
pauses briefly: the split line appears, both halves show the share of the
remaining samples they receive, and the half that gets more samples flashes
once.
"""

import json
import math

import numpy as np
import streamlit.components.v1 as components

MAX_DISPLAY_POINTS = 40_000  # per panel; larger runs are thinned evenly


def _round(a, digits=4):
    return np.round(np.asarray(a, dtype=float), digits).tolist()


def build_payload(mc, events, func, binary, reference, mc_estimate, ms_estimate):
    n = int(sum(len(e["points"]) for e in events if e["type"] != "split"))
    stride = max(1, math.ceil(n / MAX_DISPLAY_POINTS))

    xs, ys, vs, kinds, decisions, leaves = [], [], [], [], [], []
    count = 0  # evaluations so far (full resolution)
    for e in events:
        if e["type"] == "split":
            (x0, x1), (y0, y1) = e["box"]
            lb, rb = e["budget"]
            decisions.append({
                "at": count // stride, "depth": e["depth"], "dim": e["dimension"],
                "c": e["coordinate"], "box": [x0, x1, y0, y1],
                "std": [e["std"][0], e["std"][1]],
                "share": [lb / (lb + rb), rb / (lb + rb)],
            })
            continue
        pts, vals = e["points"], e["values"]
        if e["type"] == "leaf":
            (x0, x1), (y0, y1) = e["box"]
            leaves.append({"start": count // stride,
                           "end": (count + len(pts)) // stride,
                           "box": [x0, x1, y0, y1]})
        keep = (np.arange(count, count + len(pts)) % stride) == 0
        xs += _round(pts[keep, 0]); ys += _round(pts[keep, 1])
        vs += _round(vals[keep], 5)
        kinds += [0 if e["type"] == "pilot" else 1] * int(keep.sum())
        count += len(pts)

    mpts, mvals = mc["points"][::stride], mc["values"][::stride]

    g = np.linspace(0.0, 1.0, 120)
    gx, gy = np.meshgrid(g, g)
    heat = func(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    vmax = float(max(np.max(np.abs(heat)), max(vs, default=0.0),
                     float(np.max(mvals)) if len(mvals) else 0.0, 1e-12))

    return {
        "n": len(xs), "stride": stride, "binary": bool(binary), "vmax": vmax,
        "reference": reference, "mcEstimate": mc_estimate, "msEstimate": ms_estimate,
        "heat": _round(heat / vmax, 3),
        "mc": {"x": _round(mpts[:, 0]), "y": _round(mpts[:, 1]), "v": _round(mvals, 5)},
        "ms": {"x": xs, "y": ys, "v": vs, "kind": kinds},
        "decisions": decisions, "leaves": leaves,
    }


def render(payload, height=640):
    html = _TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    components.html(html, height=height, scrolling=False)


_TEMPLATE = r"""
<!doctype html><html><head><meta charset="utf-8"><style>
 body{margin:0;padding:8px;font-family:"Source Sans Pro","Segoe UI",sans-serif;color:#262730;background:#ffffff;border-radius:8px}
 .bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:13px;margin:2px 0 8px}
 .bar button{background:#ff4b4b;color:#fff;border:0;border-radius:6px;padding:5px 12px;cursor:pointer;font-size:13px}
 .cols{display:grid;grid-template-columns:1fr 1fr;gap:18px}
 .head{display:flex;justify-content:space-between;align-items:baseline}
 .head h4{margin:0 0 2px;font-size:17px}
 .stat{font-size:12.5px;color:#555;min-height:34px;font-variant-numeric:tabular-nums}
 .wrap{position:relative;width:100%;aspect-ratio:1/1;max-width:520px;margin:auto}
 canvas{position:absolute;left:0;top:0;width:100%;height:100%}
 .legend{font-size:11.5px;color:#555;text-align:center;margin-top:4px}
 .legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 4px 0 10px;vertical-align:middle}
</style></head><body>
<div class="bar">
 <button id="replay">▶ Replay</button>
 <label>Speed <select id="speed"><option value="8">8 s</option><option value="15" selected>15 s</option><option value="25">25 s</option></select></label>
 <label><input type="checkbox" id="pause" checked> Pause at MISER decisions (depth ≤
  <select id="pdepth"><option>1</option><option>2</option><option selected>3</option><option>12</option></select>)</label>
</div>
<div class="cols">
 <div><div class="head"><h4 style="color:#1f77b4">Plain Monte Carlo</h4></div>
  <div class="stat" id="mcstat"></div><div class="wrap" id="mcw"></div>
  <div class="legend" id="mcleg"></div></div>
 <div><div class="head"><h4 style="color:#ff7f0e">MISER</h4></div>
  <div class="stat" id="msstat"></div><div class="wrap" id="msw"></div>
  <div class="legend" id="msleg"></div></div>
</div>
<script>
const D = __PAYLOAD__;
const RGB = {mc:"31,119,180", ms:"255,127,14"};
const $ = id => document.getElementById(id);
const fmt = (x,d=6) => Number.isFinite(x) ? x.toFixed(d) : "—";
const P = {};

function layer(wrap){const c=document.createElement("canvas");wrap.appendChild(c);return c;}
function setup(key){
  const w=$(key+"w"); w.innerHTML="";
  const L={heat:layer(w),pts:layer(w),lines:layer(w),top:layer(w)};
  const dpr=window.devicePixelRatio||1, W=w.clientWidth, pad=4, S=W-2*pad;
  for(const c of Object.values(L)){c.width=W*dpr;c.height=W*dpr;const x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);}
  P[key]={L,px:x=>pad+x*S,py:y=>pad+(1-y)*S,S,pad,W};
  // background: faint map of f (darker = larger f)
  const h=L.heat.getContext("2d"), k=D.heat.length, cs=S/k;
  for(let j=0;j<k;j++)for(let i=0;i<k;i++){const v=Math.max(0,D.heat[j][i]);if(v<0.01)continue;
    h.fillStyle=`rgba(90,90,90,${0.07+0.33*v})`;h.fillRect(pad+i*cs,pad+(k-1-j)*cs,cs+0.6,cs+0.6);}
  h.strokeStyle="#888";h.lineWidth=1;h.strokeRect(pad,pad,S,S);
}
function clear(c){const x=c.getContext("2d");x.save();x.setTransform(1,0,0,1,0,0);x.clearRect(0,0,c.width,c.height);x.restore();}
function color(key,v,kind){
  if(kind===0) return "rgba(122,60,170,0.30)";             // MISER pilot sample (faint)
  if(D.binary) return v? `rgba(${RGB[key]},0.85)` : "rgba(150,160,175,0.45)";
  const t=Math.max(0,Math.min(1,v/D.vmax));
  return t<0.05? "rgba(150,160,175,0.45)" : `rgba(${RGB[key]},${0.25+0.7*t})`;
}
function paint(key,src,from,to){
  const p=P[key],x=p.L.pts.getContext("2d"),r=Math.max(1.1,Math.min(2.4,7000/Math.max(D.n,1)));
  for(let i=from;i<to;i++){x.fillStyle=color(key,src.v[i],src.kind?src.kind[i]:1);
    const big=D.binary&&src.v[i]&&!(src.kind&&src.kind[i]===0);   // f = 1 samples drawn larger
    x.beginPath();x.arc(p.px(src.x[i]),p.py(src.y[i]),big?r*1.8:r,0,6.283);x.fill();}
}
function splitLine(ctx,p,d,w,col){
  const[b0,b1,b2,b3]=d.box;ctx.strokeStyle=col;ctx.lineWidth=w;ctx.beginPath();
  if(d.dim===0){ctx.moveTo(p.px(d.c),p.py(b2));ctx.lineTo(p.px(d.c),p.py(b3));}
  else{ctx.moveTo(p.px(b0),p.py(d.c));ctx.lineTo(p.px(b1),p.py(d.c));}ctx.stroke();
}
function halves(d){const[x0,x1,y0,y1]=d.box;return d.dim===0?
  [[x0,d.c,y0,y1],[d.c,x1,y0,y1]]:[[x0,x1,y0,d.c],[x0,x1,d.c,y1]];}
function drawTop(t){
  const p=P.ms,c=p.L.top,x=c.getContext("2d");clear(c);
  if(S.leaf>=0){const b=D.leaves[S.leaf].box;x.setLineDash([4,3]);x.strokeStyle="#c2410c";x.lineWidth=1.4;
    x.strokeRect(p.px(b[0]),p.py(b[3]),p.px(b[1])-p.px(b[0]),p.py(b[2])-p.py(b[3]));x.setLineDash([]);}
  if(S.paused>=0){
    const d=D.decisions[S.paused],H=halves(d),big=d.share[0]>=d.share[1]?0:1;
    const flash=0.5*Math.max(0,1-(t-S.pauseStart)/PAUSE_MS);   // one short flash, fading out
    H.forEach((h,i)=>{x.fillStyle=i===big?`rgba(255,127,14,${flash})`:"rgba(120,120,120,0.08)";
      x.fillRect(p.px(h[0]),p.py(h[3]),p.px(h[1])-p.px(h[0]),p.py(h[2])-p.py(h[3]));});
    splitLine(x,p,d,3,"#7f3f00");
    H.forEach((h,i)=>{const cx=p.px((h[0]+h[1])/2),cy=p.py((h[2]+h[3])/2);
      const s=`${Math.round(100*d.share[i])}%`;
      x.font=(i===big?"bold ":"")+"13px sans-serif";x.textAlign="center";
      x.fillStyle="rgba(255,255,255,0.85)";const w=x.measureText(s).width+8;x.fillRect(cx-w/2,cy-11,w,16);
      x.fillStyle=i===big?"#7f3f00":"#555";x.fillText(s,cx,cy+1);});
  }
}
let S=null, raf=null;
const PAUSE_MS=700;   // short stop at each decision
function start(){
  if(raf)cancelAnimationFrame(raf);
  setup("mc");setup("ms");
  S={k:0,prog:0,last:null,dec:0,leaf:-1,paused:-1,pauseUntil:0,mcSum:0,done:false};
  $("mcleg").innerHTML=D.binary?'<i style="background:#1f77b4"></i>f = 1<i style="background:#96a0af"></i>f = 0':
    '<i style="background:#1f77b4"></i>darker = larger f<i style="background:#96a0af"></i>f ≈ 0';
  $("msleg").innerHTML=(D.binary?'<i style="background:#ff7f0e"></i>f = 1<i style="background:#96a0af"></i>f = 0':
    '<i style="background:#ff7f0e"></i>darker = larger f')+'<i style="background:#c4abdc"></i>pilot sample';
  raf=requestAnimationFrame(tick);
}
function tick(t){
  const seconds=+$("speed").value, pauseOn=$("pause").checked, pd=+$("pdepth").value;
  if(S.paused>=0){ if(t<S.pauseUntil){drawTop(t);stats();raf=requestAnimationFrame(tick);return;}
    splitLine(P.ms.L.lines.getContext("2d"),P.ms,D.decisions[S.paused],1.2,"rgba(160,82,45,0.7)");S.paused=-1;S.last=t;}
  if(S.last!==null)S.prog+=Math.min(t-S.last,250)/1000*D.n/seconds;S.last=t;
  let to=Math.min(D.n,Math.floor(S.prog));
  // stop exactly at the next decision
  while(S.dec<D.decisions.length&&D.decisions[S.dec].at<=to){
    const d=D.decisions[S.dec];
    if(pauseOn&&d.depth<=pd){to=Math.max(S.k,d.at);S.prog=to;S.paused=S.dec;S.pauseStart=t;S.pauseUntil=t+PAUSE_MS;S.dec++;break;}
    splitLine(P.ms.L.lines.getContext("2d"),P.ms,d,Math.max(0.6,1.4-0.15*d.depth),"rgba(160,82,45,0.55)");S.dec++;
  }
  if(to>S.k){
    for(let i=S.k;i<to;i++)S.mcSum+=D.mc.v[i];
    paint("mc",D.mc,S.k,to);paint("ms",D.ms,S.k,to);S.k=to;
    S.leaf=D.leaves.findIndex(l=>l.start<=S.k&&S.k<l.end);
  }
  if(S.k>=D.n&&S.paused<0){S.done=true;S.leaf=-1;}
  drawTop(t);stats();
  if(!S.done)raf=requestAnimationFrame(tick);
}
function stats(){
  const k=S.k,n=D.n,pct=(100*k/Math.max(n,1)).toFixed(0);
  $("mcstat").innerHTML=`Samples ${pct}% · running estimate <b>${fmt(S.done?D.mcEstimate:(k?S.mcSum/k:NaN))}</b> · exact ${fmt(D.reference)}`;
  const shown=D.decisions.filter((d,i)=>i<S.dec).length;
  $("msstat").innerHTML=`Samples ${pct}% · decisions ${shown}/${D.decisions.length}`+
    (S.done?` · estimate <b>${fmt(D.msEstimate)}</b>`:"");
}
$("replay").onclick=start;
window.addEventListener("resize",()=>{clearTimeout(window._rt);window._rt=setTimeout(start,200);});
start();
</script></body></html>
"""
