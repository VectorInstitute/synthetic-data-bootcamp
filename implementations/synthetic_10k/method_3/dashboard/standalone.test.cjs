const {test}=require('node:test');
const assert=require('node:assert/strict');
const {gzipSync}=require('node:zlib');
const {createProvider}=require('./standalone.js');
function fixture(){
 const sectors=['A','B'],pools={};
 for(const sector of sectors){
  const methods=[1,2,3].map(m=>({id:`method_${m}`,label:`Method ${m}`,rows:Array.from({length:40},(_,i)=>({id:`r${i}`,sector,metrics:{revenue:i},checks:[],statements:{units:'USD'}})),diagnostics:{attempted:45,accepted:40,rejected:5,adjusted:null,adjustment_note:'Original adjustments'}}));
  pools[sector]=gzipSync(JSON.stringify({methods})).toString('base64');
 }
 return {schema_version:1,pool_size:40,source_generation_seed:42,pools,shared:{real:[{id:'realA',sector:'A'},{id:'realB',sector:'B'}],sectors,model_info:{A:{n:1},B:{n:1}},provenance:{eligible:2},limitations:[]}};
}
test('offline controls resample fixed pools with matched sector counts',async()=>{
 const p=createProvider(fixture()),a=await p.generate(20,7,'all'),b=await p.generate(20,7,'all');
 assert.deepEqual(a,b);
 assert.equal(a.methods.length,3);
 for(const m of a.methods){assert.equal(m.rows.length,20);assert.equal(new Set(m.rows.map(r=>r.id)).size,20);assert.equal(m.diagnostics.accepted,80);}
 assert.deepEqual(a.methods[0].rows.map(r=>r.sector),a.methods[2].rows.map(r=>r.sector));
 assert.deepEqual(a.real,(await p.generate(30,7,'all')).real);
 assert.ok(a.limitations.some(s=>s.includes('fixed embedded pools')));
 assert.notDeepEqual(a.methods,(await p.generate(20,8,'all')).methods);
});
test('sector filter and sample bounds are explicit',async()=>{
 const p=createProvider(fixture()),a=await p.generate(40,0,'B');
 assert.equal(a.real.length,1);
 assert.ok(a.methods.every(m=>m.rows.every(r=>r.sector==='B')));
 await assert.rejects(()=>p.generate(41,0,'B'));
 await assert.rejects(()=>p.generate(20,0,'missing'));
 await assert.rejects(()=>p.generate(20,-1,'A'));
});
