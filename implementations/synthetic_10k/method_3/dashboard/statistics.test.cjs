const {test}=require('node:test');
const assert=require('node:assert/strict');
const {correlation,ks,values,quantile}=require('./statistics.js');
test('pairwise correlation uses complete pairs and reports constants unavailable',()=>{
 assert.deepEqual(correlation([{a:1,b:2},{a:2,b:4},{a:null,b:9},{a:3,b:6}], 'a','b'),{r:1,n:3});
 assert.equal(correlation([{a:1,b:2},{a:1,b:3},{a:1,b:4}],'a','b').r,null);
});
test('KS handles ties and distinct populations without treating null as zero',()=>{
 assert.equal(ks([1,1,2],[1,1,2]),0);
 assert.equal(ks([1,2],[3,4]),1);
 assert.equal(ks([], [1]),null);
 assert.deepEqual(values([{metrics:{a:0}},{metrics:{a:null}}],'a'),[0]);
 assert.equal(quantile([1,2,3],.5),2);
});
