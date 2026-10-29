/* Browser/Node evaluation. Formula-aligned diagnostics, not an SDMetrics runtime. */
(function(root){
 'use strict';
 const S=typeof module!=='undefined'?require('./statistics.js'):root.FinancialStats;
 const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:null;
 function pairTrend(real,synthetic,a,b){
  function fit(rows){
   const points=rows.map(x=>x.metrics).filter(x=>S.finite(x[a])&&S.finite(x[b]));
   const correlation=S.correlation(points,a,b),mx=mean(points.map(x=>x[a])),my=mean(points.map(x=>x[b]));
   const variance=points.reduce((s,x)=>s+(x[a]-mx)**2,0);
   const slope=points.length>=2&&variance>1e-20?points.reduce((s,x)=>s+(x[a]-mx)*(x[b]-my),0)/variance:null;
   const xs=points.map(x=>x[a]);
   return {...correlation,slope,intercept:slope===null?null:my-slope*mx,extent:xs.length?[Math.min(...xs),Math.max(...xs)]:null};
  }
  const r=fit(real),s=fit(synthetic);
  return {real:r,synthetic:s,score:r.r===null||s.r===null?null:1-Math.abs(r.r-s.r)/2};
 }
 function quality(real,synthetic,features){
  const columns=features.map(key=>{const r=S.values(real,key),s=S.values(synthetic,key),d=S.ks(r,s);return {key,realN:r.length,syntheticN:s.length,score:d===null?null:1-d};});
  const r=real.map(x=>x.metrics),s=synthetic.map(x=>x.metrics),pairs=[];
  features.forEach((a,i)=>features.slice(i+1).forEach(b=>{
   const rc=S.correlation(r,a,b),sc=S.correlation(s,a,b);
   pairs.push({a,b,realR:rc.r,syntheticR:sc.r,realN:rc.n,syntheticN:sc.n,score:rc.r===null||sc.r===null?null:1-Math.abs(rc.r-sc.r)/2});
  }));
  return {columnShapes:mean(columns.map(x=>x.score).filter(S.finite)),pairTrends:mean(pairs.map(x=>x.score).filter(S.finite)),columns,pairs};
 }
 function distanceModel(real,features){
  const ranges={},active=[];
  for(const key of features){const v=S.values(real,key);if(v.length>=2&&v.at(-1)>v[0]){ranges[key]=[v[0],v.at(-1)-v[0]];active.push(key);}}
  return {features:active,excluded:features.filter(k=>!active.includes(k)),ranges,real:real.map(x=>x.metrics)};
 }
 function dcr(model,rows,leaveOneOut=false){
  if(!model.features.length||!model.real.length||(leaveOneOut&&model.real.length<2))return [];
  return rows.map((row,i)=>{
   let best=Infinity;
   model.real.forEach((r,j)=>{
    if(leaveOneOut&&i===j)return;
    let total=0;
    for(const key of model.features){const a=row.metrics[key],b=r[key],af=S.finite(a),bf=S.finite(b);total+=af&&bf?Math.abs(a-b)/model.ranges[key][1]:af===bf?0:1;}
    best=Math.min(best,total/model.features.length);
   });
   return best;
  });
 }
 function summary(distances){const sorted=[...distances].sort((a,b)=>a-b);return {distances,median:S.quantile(sorted,.5),p05:S.quantile(sorted,.05),minimum:sorted[0]??null,zeroFraction:sorted.length?sorted.filter(x=>x===0).length/sorted.length:null};}
 function privacy(real,methods,features,seed){
  const model=distanceModel(real,features);let state=seed>>>0;
  const random=()=>{state=(state+0x6d2b79f5)|0;let t=Math.imul(state^state>>>15,1|state);t^=t+Math.imul(t^t>>>7,61|t);return ((t^t>>>14)>>>0)/4294967296;};
  // One shared uniform numeric baseline; equal sample counts use exactly the same rows.
  const count=Math.max(0,...methods.map(x=>x.length));
  const baseline=Array.from({length:count},()=>({metrics:Object.fromEntries(model.features.map(k=>[k,model.ranges[k][0]+random()*model.ranges[k][1]]))}));
  const baselineDistances=dcr(model,baseline);
  return {features:model.features,excluded:model.excluded,reference:summary(dcr(model,real,true)),methods:methods.map(rows=>{
   const result=summary(dcr(model,rows)),baselineMedian=summary(baselineDistances.slice(0,rows.length)).median;
   return {...result,baselineMedian,score:result.median!==null&&baselineMedian>0?Math.min(1,result.median/baselineMedian):null};
  })};
 }
 const api={quality,pairTrend,distanceModel,dcr,privacy};
 if(typeof module!=='undefined')module.exports=api;else root.FinancialEvaluation=api;
})(typeof window!=='undefined'?window:globalThis);
