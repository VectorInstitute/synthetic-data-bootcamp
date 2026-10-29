const {test}=require('node:test');
const assert=require('node:assert/strict');
const E=require('./evaluation.js');
const rows=a=>a.map(metrics=>({metrics}));
test('pair visualization fits complete observations and uses the same correlation score',()=>{
 const real=rows([{x:0,y:1},{x:1,y:3},{x:2,y:5},{x:null,y:99}]);
 const synthetic=rows([{x:0,y:5},{x:1,y:3},{x:2,y:1}]);
 const pair=E.pairTrend(real,synthetic,'x','y');
 assert.equal(pair.score,0); assert.equal(pair.real.n,3);
 assert.equal(pair.real.slope,2); assert.equal(pair.real.intercept,1);
 assert.equal(pair.synthetic.slope,-2); assert.deepEqual(pair.real.extent,[0,2]);
 assert.equal(E.pairTrend(real,rows([{x:1,y:0},{x:1,y:1},{x:1,y:2}]),'x','y').synthetic.slope,null);
 assert.equal(E.pairTrend([],synthetic,'x','y').score,null);
 assert.equal(E.pairTrend(real,rows([{x:0,y:7},{x:1,y:7},{x:2,y:7}]),'x','y').synthetic.slope,0);
});
test('KSComplement and numeric Column Pair Trends follow SDMetrics formulas',()=>{
 const real=rows([{a:0,b:0},{a:1,b:1},{a:2,b:2}]);
 const same=E.quality(real,real,['a','b']);
 assert.equal(same.columnShapes,1); assert.equal(same.pairTrends,1);
 const opposite=E.quality(real,rows([{a:0,b:2},{a:1,b:1},{a:2,b:0}]),['a','b']);
 assert.equal(opposite.columnShapes,1); assert.equal(opposite.pairTrends,0);
 const missing=E.quality(real,rows([{a:null,b:0},{a:1,b:1},{a:1,b:2}]),['a','b']);
 assert.equal(missing.pairTrends,null); assert.equal(missing.pairs[0].syntheticN,2);
 assert.equal(E.quality(real,[],['a','b']).columnShapes,null);
});
test('DCR uses real-range normalization, excludes constant features and supports missing values',()=>{
 const real=rows([{a:0,b:5},{a:10,b:5}]);
 const model=E.distanceModel(real,['a','b']);
 assert.deepEqual(model.features,['a']);
 assert.deepEqual(E.dcr(model,rows([{a:0},{a:5},{a:null},{a:20}])),[0,.5,1,1]);
 assert.deepEqual(E.dcr(model,real,true),[1,1]);
 assert.deepEqual(E.dcr(E.distanceModel(rows([{a:0},{a:10000}]),['a']),rows([{a:5000}])),[.5]);
 assert.equal(E.distanceModel(real,['b']).features.length,0);
});
test('privacy baseline is deterministic, copied rows score zero, degenerate references unavailable',()=>{
 const real=rows([{a:0},{a:1},{a:2}]);
 const a=E.privacy(real,[real],['a'],42);
 assert.deepEqual(a,E.privacy(real,[real],['a'],42));
 assert.equal(a.methods[0].score,0); assert.equal(a.methods[0].zeroFraction,1);
 assert.deepEqual(a.reference.distances,[.5,.5,.5]);
 assert.equal(E.privacy(rows([{a:2}]),[real],['a'],42).methods[0].score,null);
});
