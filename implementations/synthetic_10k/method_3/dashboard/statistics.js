/* Statistics use complete observed pairs; null never becomes zero. */
(function(root){
 const finite=x=>typeof x==='number'&&Number.isFinite(x);
 function values(rows,key){return rows.map(r=>r.metrics[key]).filter(finite).sort((a,b)=>a-b);}
 function quantile(sorted,p){if(!sorted.length)return null;const t=(sorted.length-1)*p,i=Math.floor(t);return sorted[i]+(sorted[Math.ceil(t)]-sorted[i])*(t-i);}
 function correlation(rows,a,b){const pairs=rows.filter(r=>finite(r[a])&&finite(r[b])),n=pairs.length;if(n<3)return {r:null,n};const ma=pairs.reduce((s,r)=>s+r[a],0)/n,mb=pairs.reduce((s,r)=>s+r[b],0)/n;let va=0,vb=0,c=0;for(const r of pairs){const x=r[a]-ma,y=r[b]-mb;va+=x*x;vb+=y*y;c+=x*y;}return {r:va>1e-20&&vb>1e-20?Math.max(-1,Math.min(1,c/Math.sqrt(va*vb))):null,n};}
 function ks(a,b){if(!a.length||!b.length)return null;const x=[...a].sort((a,b)=>a-b),y=[...b].sort((a,b)=>a-b);let i=0,j=0,result=0;while(i<x.length||j<y.length){const t=Math.min(i<x.length?x[i]:Infinity,j<y.length?y[j]:Infinity);while(i<x.length&&x[i]<=t)i++;while(j<y.length&&y[j]<=t)j++;result=Math.max(result,Math.abs(i/x.length-j/y.length));}return result;}
 const api={finite,values,quantile,correlation,ks};if(typeof module!=='undefined')module.exports=api;else root.FinancialStats=api;
})(typeof window!=='undefined'?window:globalThis);
