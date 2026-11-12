"""Training-progress logging: every trainer log/eval event -> metrics.jsonl + train.log + self-contained charts.html.

The HTML has no external deps (inline canvas JS) and auto-reloads every 30s, so it can be opened from the host
(mounted volume) while training runs.
"""
from __future__ import annotations

import json
import os
import time

from transformers import TrainerCallback

_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>%(title)s</title>
<meta http-equiv="refresh" content="30">
<style>body{font-family:sans-serif;margin:16px;background:#fafafa}h2{margin:4px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:14px}
.card{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:2px 6px;text-align:right}</style>
</head><body><h2>%(title)s</h2><div id="st"></div><div class="grid" id="g"></div><h3>Eval history</h3><div id="t"></div>
<script>
const D=%(data)s, S=%(status)s;
document.getElementById('st').innerHTML=`step <b>${S.step}</b> / ${S.max_steps} &middot; epoch ${(+S.epoch).toFixed(2)}
 &middot; elapsed ${(S.elapsed/60).toFixed(1)} min &middot; ETA ${S.eta_min==null?'?':S.eta_min.toFixed(1)} min
 &middot; updated ${S.updated} &middot; ${S.done?'<b>DONE</b>':'running'}`;
const charts=[["loss",["loss","eval_loss"]],["token accuracy",["mean_token_accuracy","eval_mean_token_accuracy"]],
 ["learning rate",["learning_rate"]],["grad norm",["grad_norm"]],["entropy",["entropy","eval_entropy"]],
 ["GRPO reward (mean over rollouts)",["reward","rewards/slot_f1_reward/mean","rewards/extraction_reward/mean","rewards/format_reward/mean"]],
 ["GRPO reward std / zero-std groups",["reward_std","frac_reward_zero_std"]],
 ["completion length (tokens)",["completions/mean_length","completions/max_length"]],
 ["truncated completions",["completions/clipped_ratio"]],["KL",["kl"]]];
const col=["#1f77b4","#d62728","#2ca02c","#ff7f0e"];
for(const [name,keys] of charts){
 const ser=keys.map(k=>D.filter(r=>r[k]!=null).map(r=>[r.step,r[k]])).filter(s=>s.length);
 if(!ser.length)continue;
 const c=document.createElement('div');c.className='card';
 c.innerHTML=`<b>${name}</b> `+keys.map((k,i)=>`<span style="color:${col[i]}">&#9632; ${k}</span>`).join(' ');
 const cv=document.createElement('canvas');cv.width=500;cv.height=260;c.appendChild(cv);
 document.getElementById('g').appendChild(c);
 const x=cv.getContext('2d'),all=ser.flat(),P=45;
 let x0=Math.min(...all.map(p=>p[0])),x1=Math.max(...all.map(p=>p[0])),y0=Math.min(...all.map(p=>p[1])),y1=Math.max(...all.map(p=>p[1]));
 if(x1==x0)x1=x0+1;if(y1==y0){y1+=Math.abs(y1)*.1+1e-9;y0-=Math.abs(y0)*.1+1e-9}
 const X=v=>P+(v-x0)/(x1-x0)*(cv.width-P-10),Y=v=>cv.height-25-(v-y0)/(y1-y0)*(cv.height-35);
 x.strokeStyle='#ccc';x.fillStyle='#555';x.font='10px sans-serif';
 for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4,yy=Y(v);x.beginPath();x.moveTo(P,yy);x.lineTo(cv.width-10,yy);x.stroke();
  x.fillText(Math.abs(v)<1e-3&&v!=0?v.toExponential(1):v.toPrecision(3),2,yy+3)}
 for(let i=0;i<=4;i++){const v=x0+(x1-x0)*i/4;x.fillText(Math.round(v),X(v)-8,cv.height-8)}
 keys.forEach((k,i)=>{const s=D.filter(r=>r[k]!=null).map(r=>[r.step,r[k]]);if(!s.length)return;
  x.strokeStyle=col[i];x.fillStyle=col[i];x.lineWidth=1.6;x.beginPath();
  s.forEach((p,j)=>j?x.lineTo(X(p[0]),Y(p[1])):x.moveTo(X(p[0]),Y(p[1])));x.stroke();
  if(k.startsWith('eval_'))s.forEach(p=>{x.beginPath();x.arc(X(p[0]),Y(p[1]),3,0,7);x.fill()})});
}
const E=D.filter(r=>r.eval_loss!=null||r.eval_reward!=null);
if(E.length){const ks=Object.keys(E[0]).filter(k=>k.startsWith('eval_')&&!/runtime|per_second/.test(k));
 document.getElementById('t').innerHTML='<table><tr><th>step</th><th>epoch</th>'+ks.map(k=>`<th>${k}</th>`).join('')+'</tr>'+
 E.map(r=>`<tr><td>${r.step}</td><td>${(+r.epoch).toFixed(2)}</td>`+ks.map(k=>`<td>${r[k]==null?'':(+r[k]).toFixed(4)}</td>`).join('')+'</tr>').join('')+'</table>'}
else document.getElementById('t').textContent='(no eval yet)';
</script></body></html>"""


class ProgressCallback(TrainerCallback):
    def __init__(self, log_dir: str, title: str = "training"):
        os.makedirs(log_dir, exist_ok=True)
        self.dir, self.title, self.rows, self.t0 = log_dir, title, [], time.time()
        self.jsonl = os.path.join(log_dir, "metrics.jsonl")
        open(self.jsonl, "w").close()

    def _write(self, state, done=False):
        el = time.time() - self.t0
        eta = (el / state.global_step * (state.max_steps - state.global_step) / 60) if state.global_step else None
        status = {"step": state.global_step, "max_steps": state.max_steps, "epoch": state.epoch or 0, "elapsed": el,
                  "eta_min": eta, "updated": time.strftime("%H:%M:%S"), "done": done}
        html = _HTML % {"title": self.title, "data": json.dumps(self.rows), "status": json.dumps(status)}
        tmp = os.path.join(self.dir, ".charts.html.tmp")
        with open(tmp, "w") as f:
            f.write(html)
        os.replace(tmp, os.path.join(self.dir, "charts.html"))

    def on_log(self, args, state, control, logs=None, **kw):
        if not state.is_world_process_zero or not logs:
            return
        row = {"step": state.global_step, "epoch": round(state.epoch or 0, 4), "time": round(time.time() - self.t0, 1),
               **{k: v for k, v in logs.items() if isinstance(v, (int, float))}}
        self.rows.append(row)
        with open(self.jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
        with open(os.path.join(self.dir, "train.log"), "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] step {state.global_step}/{state.max_steps} "
                    + " ".join(f"{k}={v:.4g}" for k, v in row.items() if k not in ("step", "time")) + "\n")
        self._write(state)

    def on_train_end(self, args, state, control, **kw):
        if state.is_world_process_zero:
            self._write(state, done=True)
