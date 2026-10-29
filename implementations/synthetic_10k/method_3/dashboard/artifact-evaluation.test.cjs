/* Exercise the actual delivered HTML and embedded source pools without a browser. */
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const E=require('./evaluation.js');
const {createProvider}=require('./standalone.js');
const file=path.join(__dirname,'../../dashboard/financial_method_comparison.html');
test('delivered standalone embeds current metrics and evaluates real pools at slider bounds',async()=>{
 const page=fs.readFileSync(file,'utf8').replace(/\r\n/g,'\n');
 assert.ok(page.includes(fs.readFileSync(path.join(__dirname,'evaluation.js'),'utf8').replace(/\r\n/g,'\n')));
 assert.ok(!page.includes("fetch('/api/"));
 for(const id of ['syntheticQuality','realQuality','syntheticPrivacy','realPrivacy','trendPair','syntheticTrends','realTrends'])assert.ok(page.includes(`id="${id}"`));
 assert.ok(page.includes('function drawPairTrends()'));
 assert.ok(page.includes("$('trendPair').addEventListener('change'"));
 const scripts=[...page.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)];
 const data=scripts.find(x=>x[0].startsWith('<script type="application/json"'));
 const snapshot=JSON.parse(data[1]);
 for(const script of scripts)if(script!==data)new vm.Script(script[1]);
 const provider=createProvider(snapshot),features=['revenue','asset_turnover','operating_margin','equity_ratio','cash_ratio'];
 for(const sector of ['all',...snapshot.shared.sectors]){
  for(const n of [20,2000]){
   const selected=await provider.generate(n,42,sector);
   const result=E.privacy(selected.real,selected.methods.map(m=>m.rows),features,42);
   assert.ok(result.features.length>0);
   result.methods.forEach((p,i)=>{
    assert.equal(p.distances.length,n);
    assert.ok(p.distances.every(d=>Number.isFinite(d)&&d>=0));
    assert.ok(p.score===null||p.score>=0&&p.score<=1);
    const q=E.quality(selected.real,selected.methods[i].rows,features);
    assert.equal(q.columns.length,5);assert.equal(q.pairs.length,10);
    assert.ok(q.columnShapes===null||q.columnShapes>=0&&q.columnShapes<=1);
    assert.ok(q.pairTrends===null||q.pairTrends>=0&&q.pairTrends<=1);
    for(const p of q.pairs){
     const trend=E.pairTrend(selected.real,selected.methods[i].rows,p.a,p.b);
     assert.equal(trend.score,p.score);
     for(const fit of [trend.real,trend.synthetic])if(fit.slope!==null){
      assert.ok(Number.isFinite(fit.slope));assert.ok(Number.isFinite(fit.intercept));
     }
    }
   });
   assert.equal(result.reference.distances.length,selected.real.length);
  }
 }
});
