"""Atomic, offline-capable training dashboard; no external JS dependencies."""
import json, math, time
from pathlib import Path

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

HTML='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scene token 训练监控</title>
<style>body{margin:0;background:#f3f5f8;color:#182334;font:15px system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:28px}h1{margin:0 0 8px}p{line-height:1.6}.card{background:white;border:1px solid #dce2e9;border-radius:10px;padding:18px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}canvas{width:100%;height:240px}.status{font-weight:650}select,button{padding:9px;margin:4px;border:1px solid #b8c3d2;border-radius:5px}img{width:100%;image-rendering:auto}.imgs{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}small{color:#617187}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:9px;text-align:left}.warn{color:#a84416}@media(max-width:800px){.grid,.imgs{grid-template-columns:1fr}}</style>
<main><h1>Scene token 训练监控</h1><p>S=256 · 初始化 std=0.02 · DL3DV · 252px · 每 15 秒刷新状态<br><small>固定验证场景从训练集按场景隔离。训练视角数变化时训练曲线不构成同难度比较，请优先查看固定验证。</small></p>
<div class="card"><div id="status" class="status">加载中…</div><p id="details"></p><p id="fresh"></p><a href="metrics.jsonl">完整训练日志 JSONL</a> · <a href="evaluations.json">验证记录</a> · <a href="config.json">训练配置</a> · <a href="checkpoints.json">Checkpoint 索引</a><pre id="error" class="warn"></pre></div>
<div class="grid"><div class="card">训练 Loss（每 100 步均值）<canvas id="loss"></canvas></div><div class="card">固定验证 PSNR ↑<canvas id="psnr"></canvas></div><div class="card">固定验证 LPIPS ↓<canvas id="lpips"></canvas></div></div>
<div class="card"><h2>最近是否还在进步</h2><div id="trend"></div><small>仅描述最近两次固定评估的差值，不自动判断收敛或停止训练。V2 与 V6 分开比较；更高 PSNR 与更低 LPIPS 均值得观察。</small></div>
<div class="card"><h2>重建效果对比</h2><label>当前评估<select id="step"></select></label><label>历史对照<select id="prev"></select></label><label>集合<select id="set"><option value="val_v2">验证 V2</option><option value="val_v6">验证 V6</option><option value="train_probe">训练参考 V2</option></select></label><label>场景<select id="scene"></select></label><label>目标图<select id="target"></select></label><div id="score"></div><div class="imgs"><div>目标图 GT<img id="gt"></div><div>历史重建<img id="old"></div><div>当前重建<img id="new"></div></div></div>
<div class="card"><h2>保存与异常处理</h2><p>首 50 步及每 1000 步保存完整可恢复状态；保留最近 3 份、阶段末、指定里程碑及验证最佳 checkpoint。每 1000 步评估固定场景并保存图像。状态超过 3 分钟未更新会提示检查，不会继续显示为正常。出现 NaN、跳过优化器更新、数据错误或进程退出时保留异常日志；不会自动掩盖错误后继续。</p><p>停止方法：在服务器本次运行目录创建 <code>STOP</code> 文件。训练将在安全边界保存后停止；恢复时使用管理脚本，网页是只读的。</p></div></main>
<script>
let D={evaluations:[],train:[]}, initialized=false;
const $=id=>document.getElementById(id),fmt=(x,n=4)=>Number.isFinite(x)?x.toFixed(n):'—';
function options(el,items,defaultValue){let old=el.value;el.replaceChildren(...items.map(([v,l])=>{let o=document.createElement('option');o.value=v;o.textContent=l;return o}));el.value=items.some(x=>String(x[0])===old)?old:defaultValue;}
function chart(id,series){let c=$(id),w=c.clientWidth,h=240;c.width=w*2;c.height=h*2;let g=c.getContext('2d');g.scale(2,2);g.clearRect(0,0,w,h);let all=series.flatMap(s=>s.points).filter(p=>Number.isFinite(p[1]));if(!all.length)return;let xmax=Math.max(...all.map(p=>p[0]),1),ymin=Math.min(...all.map(p=>p[1])),ymax=Math.max(...all.map(p=>p[1]));let pad=(ymax-ymin)*.12||.1;ymin-=pad;ymax+=pad;g.font='11px system-ui';for(let i=0;i<4;i++){let y=24+(h-58)*i/3;g.strokeStyle='#e4e8ef';g.beginPath();g.moveTo(46,y);g.lineTo(w-10,y);g.stroke();g.fillStyle='#607088';g.fillText(fmt(ymax-(ymax-ymin)*i/3,2),2,y+4)}series.forEach((s,i)=>{g.strokeStyle=s.color;g.lineWidth=2;g.beginPath();s.points.filter(p=>Number.isFinite(p[1])).forEach((p,j)=>{let x=46+p[0]/xmax*(w-60),y=24+(ymax-p[1])/(ymax-ymin)*(h-58);j?g.lineTo(x,y):g.moveTo(x,y)});g.stroke();g.fillStyle=s.color;g.fillText(s.name,48+i*100,h-9)});g.fillStyle='#607088';g.fillText(xmax+' 步',w-72,h-25)}
function images(){let e=D.evaluations.find(e=>String(e.step)===$('step').value),p=D.evaluations.find(e=>String(e.step)===$('prev').value),set=$('set').value;if(!e||!e.sets[set])return;let entries=e.sets[set].scenes;options($('scene'),entries.filter(r=>r.images).map(r=>[r.scene,r.scene.slice(0,12)]),entries.find(r=>r.images)?.scene);let r=entries.find(r=>r.scene===$('scene').value);if(!r)return;options($('target'),r.targets.map((v,i)=>[i,'帧 '+v]),'0');let t=Number($('target').value),old=p?.sets[set]?.scenes.find(x=>x.scene===r.scene);$('gt').src=r.images[t].gt;$('new').src=r.images[t].render;$('old').src=old?.images?.[t]?.render||r.images[t].gt;$('score').textContent='当前场景全部目标均值：PSNR '+fmt(r.psnr,3)+' · LPIPS '+fmt(r.lpips)+'；历史：PSNR '+fmt(old?.psnr,3)+' · LPIPS '+fmt(old?.lpips);}
async function refresh(){try{let response=await fetch('dashboard.json?t='+Date.now(),{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);D=await response.json();let s=D.status||{};$('status').textContent=(s.state||'准备中')+' · 阶段 '+(s.phase||'—')+' · '+(s.step||0)+' / '+(s.total_steps||50000)+' 步';$('details').textContent=s.message||('训练输入 '+(s.views||'—')+' 视角 · 剩余磁盘 '+fmt(s.free_gib,1)+' GiB · 最近每步 '+fmt(s.seconds,3)+' 秒');let age=(Date.now()/1000-(s.updated||0)),stale=age>180&&!['complete','paused','failed','phase_complete'].includes(s.state);$('fresh').textContent='最后更新 '+new Date((s.updated||0)*1000).toLocaleString()+(stale?' · 状态已过期，请检查后台进程':'');$('fresh').className=stale?'warn':'';$('error').textContent=s.error||'';
chart('loss',[{name:'训练均值',color:'#2467bd',points:D.train.map(r=>[r.step,r.loss])}]);for(let metric of ['psnr','lpips'])chart(metric,[['val_v2','#2467bd'],['val_v6','#bf6532']].map(([k,color])=>({name:k,color,points:D.evaluations.filter(r=>r.sets[k]).map(r=>[r.step,r.sets[k].mean[metric]])})));
let recent=D.evaluations.slice(-2);$('trend').textContent=recent.length<2?'至少需要两次固定评估后显示变化。':['val_v2','val_v6'].map(k=>{let a=recent[0].sets[k]?.mean,b=recent[1].sets[k]?.mean;return k+': '+recent[0].step+' → '+recent[1].step+' 步，PSNR Δ '+fmt(b?.psnr-a?.psnr,3)+' dB，LPIPS Δ '+fmt(b?.lpips-a?.lpips)}).join('； ');
let items=D.evaluations.map(e=>[e.step,e.step+' 步']);let follow=$('step').value===String(D.evaluations.at(-2)?.step);options($('step'),items,String(items.at(-1)?.[0]));options($('prev'),items,String(items.at(-2)?.[0]??items[0]?.[0]));if(follow||!initialized)$('step').value=String(items.at(-1)?.[0]);initialized=true;images();}catch(e){$('status').textContent='暂时无法读取监控数据：'+e.message;}}
for(let id of ['step','prev','set','scene','target'])$(id).addEventListener('change',images);refresh();setInterval(refresh,15000);
</script>'''

def publish(root,status=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if status is not None:atomic(root/'status.json',status)
    state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    rows=[]
    if (root/'metrics.jsonl').exists():
        for line in (root/'metrics.jsonl').read_text().splitlines():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:continue
    train=[]
    for i in range(0,len(rows),100):
        chunk=rows[i:i+100];train.append(dict(step=chunk[-1]['step'],loss=sum(r['loss'] for r in chunk)/len(chunk)))
    evaluations=json.loads((root/'evaluations.json').read_text()) if (root/'evaluations.json').exists() else []
    atomic(root/'dashboard.json',dict(status=state,train=train,evaluations=evaluations))
    if not (root/'index.html').exists():
        tmp=root/'index.html.tmp';tmp.write_text(HTML,encoding='utf-8');tmp.replace(root/'index.html')
