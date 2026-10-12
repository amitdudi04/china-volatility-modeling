"""Build a self-contained, checksum-verified OFFLINE research dashboard.

No Qt, web service, CDN, API keys, external images or finance data downloads.
The HTML file embeds only verified outputs/research_v22 data. The script
refuses overwriting an existing dashboard without explicitly changing path.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from src.gui.v22_data import MARKETS,load_research_market
from src.risk_economic_v22 import align_next_session_returns

def export_dashboard(destination:Path):
    destination=Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing overwrite: {destination}")
    payload={}
    for market in MARKETS:
        d=load_research_market(market)
        p=d["predictions"]
        r=align_next_session_returns(p,d["prices"])
        simple=np.expm1(r.to_numpy(float))
        sigma=np.sqrt(p["garch11_variance"].to_numpy(float))
        weights=np.minimum(1.,.20/sigma)
        turnover=np.abs(weights-np.r_[1.,weights[:-1]])
        datasets={
            "market":market,
            "source":d["source"],
            "scope":d["scope"],
            "note":d["instrument_note"],
            "dates":[t.strftime("%Y-%m-%d") for t in p.index],
            "realized":[round(float(x),6) for x in np.sqrt(p["realized_next_variance"])],
            "garch":[round(float(x),6) for x in np.sqrt(p["garch11_variance"])],
            "rf":[round(float(x),6) for x in np.sqrt(p["random_forest_variance"])],
            "static_wealth":[round(float(x),6) for x in np.cumprod(1.+simple)],
            "managed_wealth":[round(float(x),6) for x in np.cumprod(1.+weights*simple-.001*turnover)],
            "scores":d["model_summary"].to_dict(orient="records"),
            "bootstrap":d["bootstrap"].to_dict(orient="records"),
            "risk":d["risk"].to_dict(orient="records"),
            "strategies":d["strategy"].loc[d["strategy"]["cost_bps"]==10].to_dict(orient="records"),
        }
        payload[market]=datasets
    def sanitize(value):
        if isinstance(value,dict):
            return {key:sanitize(item) for key,item in value.items()}
        if isinstance(value,list):
            return [sanitize(item) for item in value]
        if isinstance(value,(int,float,np.integer,np.floating)):
            if not np.isfinite(value):
                return None
            return float(value) if isinstance(value,(float,np.floating)) else int(value)
        return value
    blob=json.dumps(sanitize(payload),allow_nan=False,separators=(",",":"),ensure_ascii=True).replace("</","<\\/")
    html=r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chinese Volatility Research v2.2 | Archived 2024</title>
<style>
:root{color-scheme:dark;font-family:system-ui,Arial,sans-serif}body{margin:0;background:#0d1520;color:#ecf2f8}
header{padding:25px 4vw;background:#142337;border-bottom:1px solid #324459}
h1{margin:0 0 8px;font-size:1.65rem}h2{font-size:1.2rem;margin:0 0 10px}
.banner{background:#342c22;color:#f9dab0;padding:12px 4vw;line-height:1.5}
main{max-width:1350px;padding:24px 4vw;margin:auto}
section{margin:18px 0;border:1px solid #35465a;border-radius:12px;padding:20px;background:#182536}
button,select{background:#2a455e;color:#fff;border:1px solid #7190a6;border-radius:7px;padding:10px 16px;font-size:1rem;cursor:pointer}
button.active{background:#38657c;border-color:#c6e2f2}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
nav{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0}
p{line-height:1.55;color:#d2dce9}
table{border-collapse:collapse;width:100%;font-size:.87rem}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #354459}th{color:#d7e9ff;background:#23364b}td.number{text-align:right;font-variant-numeric:tabular-nums}
.scroller{overflow-x:auto}.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.kpi{border:1px solid #35506b;border-radius:8px;padding:13px}.kpi strong{display:block;margin-bottom:7px}.muted{color:#a6bacd}
canvas{width:100%;height:320px;display:block;background:#132031;border-radius:8px}
.warning{border-left:4px solid #dda866;padding:10px 14px;background:#302e2a}
footer{padding:35px 4vw;color:#a6bacd;text-align:center}
</style></head><body>
<header><h1>Chinese Equity Volatility — Research v2.2</h1><p>Historical prequential forecasting and risk sensitivity | 2024 evaluation | original archived datasets</p></header>
<div class="banner"><strong>RESEARCH INTEGRITY:</strong> Retrospective 2024 data were examined during model development. This is <strong>not an untouched holdout</strong>, not a claim of adaptive copula modeling, and not a validated investable risk-management strategy.</div>
<main><div class="controls"><label for="market">Market</label><select id="market"></select><span id="instrument" class="muted"></span></div>
<nav id="tabs"></nav><div id="content"></div>
<section><h2>Research provenance and limitations</h2><div id="provenance"></div>
<p>Data and checksum manifests: <code>outputs/research_v22/</code>. Canonical methodology: <code>docs/RESEARCH_V22_ARCHIVED_RESULTS.md</code>. The original legacy pipeline remains a separate, superseded implementation.</p></section></main>
<footer>Self-contained offline dashboard | No external finance feeds | Generated from SHA-verified historical research files</footer>
<script id="verified-data" type="application/json"><!--DATA--></script>
<script>
const DB=JSON.parse(document.getElementById('verified-data').textContent);
const MARKETS=['CSI_300','SSE_Composite','ChiNext'];let current='CSI_300',page='Overview';
const TABS=['Overview','Forecasts','Tail risk','Economic results'];
const fmt=(x,k=4)=>typeof x==='number'&&Number.isFinite(x)?x.toFixed(k):String(x);
const pretty=s=>s.replaceAll('_',' ');
const root=document.getElementById('content');
function table(rows,fields,percentFields=[]){if(!rows?.length)return '<p>No verified observations.</p>';return '<div class="scroller"><table><thead><tr>'+fields.map(f=>'<th>'+pretty(f)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+fields.map(f=>{const x=r[f];return '<td class="'+(typeof x==='number'?'number':'')+'">'+(typeof x==='number'?(percentFields.includes(f)?fmt(x*100,2)+'%':fmt(x,4)):String(x))+'</td>';}).join('')+'</tr>').join('')+'</tbody></table></div>'}
function chart(canvas,lines,dates,ylabel){
 const dpr=window.devicePixelRatio||1,w=canvas.clientWidth||900,h=320;
 canvas.width=Math.floor(w*dpr);canvas.height=Math.floor(h*dpr);
 const c=canvas.getContext('2d');c.scale(dpr,dpr);const pad={l:70,r:25,t:27,b:42};
 const all=Object.values(lines).flat(),min=Math.min(...all),max=Math.max(...all);
 const lo=Math.min(0,min),hi=max>lo?max:lo+1;const iw=w-pad.l-pad.r,ih=h-pad.t-pad.b;
 const yp=v=>pad.t+ih*(1-(v-lo)/(hi-lo)),xp=i=>pad.l+iw*i/(dates.length-1||1);
 c.strokeStyle='#526176';c.lineWidth=1;c.fillStyle='#d3e4f3';c.font='12px system-ui';
 for(let i=0;i<=4;i++){const y=pad.t+i*ih/4;c.beginPath();c.moveTo(pad.l,y);c.lineTo(w-pad.r,y);c.stroke();c.fillText(fmt(hi-(hi-lo)*i/4,2),9,y+4)}
 c.fillText(dates[0],pad.l,h-12);c.fillText(dates[dates.length-1],w-112,h-12);
 const colors=['#f4c477','#71bff1','#91d9b0','#e8a8db'];let n=0;
 Object.entries(lines).forEach(([name,values])=>{c.strokeStyle=colors[n];c.lineWidth=1.7;c.beginPath();values.forEach((v,i)=>i?c.lineTo(xp(i),yp(v)):c.moveTo(xp(i),yp(v)));c.stroke();c.fillStyle=colors[n];c.fillRect(pad.l+n*175,6,14,6);c.fillText(name,pad.l+20+n*175,13);n++});
 c.fillStyle='#aabdd2';c.fillText(ylabel,pad.l,h-28);
}
function heading(text){return '<h2>'+text+'</h2>'}
function render(){
 const d=DB[current];document.getElementById('instrument').textContent=d.source.instrument+' | '+d.source.first+'–'+d.source.last;
 const tabs=document.getElementById('tabs');tabs.innerHTML=TABS.map(t=>'<button class="'+(t===page?'active':'')+'" data-tab="'+t+'">'+t+'</button>').join('');
 tabs.querySelectorAll('button').forEach(b=>b.onclick=()=>{page=b.dataset.tab;render()});
 if(page==='Overview'){
 const q=d.scores.slice().sort((a,b)=>a.qlike-b.qlike)[0],r=d.scores.slice().sort((a,b)=>a.rmse-b.rmse)[0];
 root.innerHTML='<section>'+heading('Research overview')+'<div class="kpis"><div class="kpi"><strong>2024 forecast origins</strong>240</div><div class="kpi"><strong>Lowest QLIKE</strong>'+pretty(q.model)+'<p>'+fmt(q.qlike)+'</p></div><div class="kpi"><strong>Lowest RMSE</strong>'+pretty(r.model)+'<p>'+fmt(r.rmse)+'</p></div></div><p>Targets use annualized next-session squared log returns; all models score the same 240 market sessions. Rankings are retrospective and loss-sensitive.</p>'+table(d.scores,['model','n','rmse','mae','qlike'])+'</section>';
 }else if(page==='Forecasts'){
 root.innerHTML='<section>'+heading('Realized proxy and forecasts — annualized volatility')+'<p>Square roots of annualized variance for visualization only. The measured next-session squared return is noisy.</p><canvas id="mainchart"></canvas></section><section>'+heading('Paired block-bootstrap QLIKE differences vs rolling baseline')+table(d.bootstrap,['model','mean_qlike_difference','ci95_low','ci95_high','ci_excludes_zero'])+'<p>Intervals are conditional on retrospective model choices and do not account for all multiple comparisons.</p></section>';
 requestAnimationFrame(()=>chart(document.getElementById('mainchart'),{'Realized':d.realized,'GARCH':d.garch,'Random Forest':d.rf},d.dates,'Annualized volatility'));
 }else if(page==='Tail risk'){
 root.innerHTML='<section>'+heading('Gaussian 95% VaR — Kupiec unconditional coverage')+'<p class="warning">Nonrejection of Kupiec coverage does NOT prove safe tail-risk forecasting. These earlier diagnostics do NOT describe Student-t ES or conditional-coverage tests.</p>'+table(d.risk,['model','observations','breaches','observed_rate','kupiec_pvalue'],['observed_rate'])+'</section>';
 }else{
 const rows=d.strategies.filter(x=>x.forecast_model==='garch11_variance'||x.strategy==='static').filter((x,i,a)=>a.findIndex(y=>y.strategy===x.strategy&&y.forecast_model===x.forecast_model)===i);
 root.innerHTML='<section>'+heading('Illustrative static vs GARCH-managed wealth, 10 bps')+'<p>Long-only, 20% volatility target, max exposure 100%, cash yield 0. Returns exclude dividends and full implementability costs.</p><canvas id="mainchart"></canvas></section><section>'+heading('Economic sensitivity at 10 bps')+table(rows,['strategy','annualized_return','annualized_vol','sharpe_zero_cash','max_drawdown','average_exposure'],['annualized_return','annualized_vol','max_drawdown','average_exposure'])+'</section>';
 requestAnimationFrame(()=>chart(document.getElementById('mainchart'),{'Static':d.static_wealth,'GARCH-managed':d.managed_wealth},d.dates,'Wealth multiple'));
 }
 document.getElementById('provenance').innerHTML='<p><strong>Instrument:</strong> '+d.source.instrument+' | <strong>Source rows:</strong> '+d.source.rows+'<br><strong>Source SHA-256:</strong> <code>'+d.source.sha256+'</code></p><p class="warning">'+d.note+' '+d.scope+'</p>';
}
const selector=document.getElementById('market');MARKETS.forEach(x=>selector.add(new Option(x,x)));selector.onchange=()=>{current=selector.value;render()};render();
window.addEventListener('resize',()=>{if(document.getElementById('mainchart'))render()});
</script></body></html>'''.replace("<!--DATA-->",blob)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(html,encoding="utf-8")
    return destination

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=export_dashboard(args.out)
    print(f"PASS: fully offline, three-market research dashboard: {result}")
if __name__=="__main__":
    main()
