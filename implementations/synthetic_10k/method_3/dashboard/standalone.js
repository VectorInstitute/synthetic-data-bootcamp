/* Standalone data adapter: resample native-generated pools without a server. */
(function(root){
 'use strict';
 function random(seed){let state=seed>>>0;return ()=>{state=(state+0x6D2B79F5)>>>0;let t=state;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};}
 async function unpack(encoded){
  if(typeof DecompressionStream==='undefined')throw new Error('This standalone file requires a browser with gzip DecompressionStream support, such as a recent Chrome, Edge, Firefox or Safari');
  const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
  const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return JSON.parse(await new Response(stream).text());
 }
 function shuffle(n,rng){const indexes=Array.from({length:n},(_,i)=>i);for(let i=n-1;i>0;i--){const j=Math.floor(rng()*(i+1));[indexes[i],indexes[j]]=[indexes[j],indexes[i]];}return indexes;}
 function aggregateDiagnostics(methods){
  const result={attempted:0,accepted:0,rejected:0,adjusted:methods.every(m=>m.diagnostics.adjusted===null)?null:0,rejection_reasons:{},adjustments_by_field:{},mean_absolute_adjustment:{}};
  const weighted={};
  for(const m of methods){const d=m.diagnostics;for(const k of ['attempted','accepted','rejected'])result[k]+=d[k];if(result.adjusted!==null)result.adjusted+=d.adjusted||0;
   for(const [k,v] of Object.entries(d.rejection_reasons||{}))result.rejection_reasons[k]=(result.rejection_reasons[k]||0)+v;
   for(const [k,v] of Object.entries(d.adjustments_by_field||{})){result.adjustments_by_field[k]=(result.adjustments_by_field[k]||0)+v;weighted[k]=(weighted[k]||0)+(d.mean_absolute_adjustment[k]||0)*v;}
   if(d.unique_templates!==undefined){result.unique_templates=(result.unique_templates||0)+d.unique_templates;result.template_reuses=(result.template_reuses||0)+d.template_reuses;}
  }
  for(const [k,v] of Object.entries(weighted))result.mean_absolute_adjustment[k]=v/result.adjustments_by_field[k];
  result.adjustment_note=`Pool-generation statistics for ${result.accepted.toLocaleString('en-US')} embedded samples, before interactive resampling. ${methods[0].diagnostics.adjustment_note||''}`;
  if(result.adjusted===null)delete result.mean_absolute_adjustment;
  return result;
 }
 function createProvider(snapshot){
  const cache=new Map(),shared=snapshot.shared;
  const load=sector=>{if(!cache.has(sector))cache.set(sector,unpack(snapshot.pools[sector]));return cache.get(sector);};
  return {
   async info(){return {sectors:shared.sectors,source:shared.provenance};},
   async generate(n,seed,sector){
    if(!Number.isInteger(n)||n<20||n>snapshot.pool_size)throw new Error(`Sample count must be an integer from 20 to ${snapshot.pool_size}`);
    if(!Number.isInteger(seed)||seed<0||seed>4294967295)throw new Error('Sampling seed must be an integer from 0 to 4294967295');
    if(sector!=='all'&&!shared.sectors.includes(sector))throw new Error('Unknown sector');
    const sectors=sector==='all'?shared.sectors:[sector];
    const real=shared.real.filter(r=>sector==='all'||r.sector===sector);
    const rng=random(seed),weights=sectors.map(s=>real.filter(r=>r.sector===s).length),total=weights.reduce((a,b)=>a+b,0),schedule=[],counts={};
    for(let i=0;i<n;i++){let s=sector;if(sector==='all'){let t=rng()*total;s=sectors[sectors.length-1];for(let j=0;j<sectors.length;j++){t-=weights[j];if(t<0){s=sectors[j];break;}}}schedule.push(s);counts[s]=(counts[s]||0)+1;}
    const loaded=await Promise.all(sectors.map(load)),bySector=Object.fromEntries(sectors.map((s,i)=>[s,loaded[i]]));
    const methods=[0,1,2].map(index=>{
     const pickRng=random((seed+Math.imul(index+1,2654435761))>>>0),indexes={},positions={};
     for(const s of sectors){indexes[s]=shuffle(bySector[s].methods[index].rows.length,pickRng);positions[s]=0;}
     const rows=schedule.map(s=>{const row=bySector[s].methods[index].rows[indexes[s][positions[s]++]];return {...row,pool_member_id:row.id,id:`${s}:${row.id}`};});
     const sourceMethods=sectors.map(s=>bySector[s].methods[index]);
     return {id:sourceMethods[0].id,label:sourceMethods[0].label,rows,diagnostics:aggregateDiagnostics(sourceMethods)};
    });
    return {...shared,real,methods,n,seed,sector,sector_counts:counts,
     model_info:Object.fromEntries(sectors.map(s=>[s,shared.model_info[s]])),
     standalone:{mode:'fixed_pool_resampling',pool_size_per_sector:snapshot.pool_size,source_generation_seed:snapshot.source_generation_seed,built_at:snapshot.built_at},
     limitations:[`Standalone snapshot: controls resample fixed embedded pools of ${snapshot.pool_size.toLocaleString('en-US')} native-generated companies per method per sector. The sampling seed changes selection, not the Python generator. Pool generation seed: ${snapshot.source_generation_seed}.`,
      'Generation diagnostics describe the embedded pools, not rejection rates for the interactive selection. Rebuild this HTML with Python to generate new pools.',...(shared.limitations||[])]};
   }
  };
 }
 if(typeof module!=='undefined'&&module.exports)module.exports={createProvider};
 else root.financialDataProvider=createProvider(JSON.parse(document.getElementById('embedded-financial-data').textContent));
})(typeof window!=='undefined'?window:globalThis);
