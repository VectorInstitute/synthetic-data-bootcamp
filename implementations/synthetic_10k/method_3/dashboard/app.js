'use strict';
const $=id=>document.getElementById(id), S=FinancialStats;
const METRICS={revenue:['Revenue','money'],total_assets:['Total assets','money'],total_liabilities:['Total liabilities','money'],stockholders_equity:['Shareholders’ equity','money'],net_income:['Net income','money'],operating_income:['Operating income','money'],operating_cash_flow:['Operating cash flow','money'],gross_margin:['Gross margin','ratio'],operating_margin:['Operating margin','ratio'],net_margin:['Net margin','ratio'],asset_turnover:['Asset turnover','times'],equity_ratio:['Equity / assets','ratio'],liabilities_ratio:['Liabilities / assets','ratio'],cash_ratio:['Cash / assets','ratio'],rd_ratio:['R&D / revenue','ratio'],ocf_margin:['Operating cash flow / revenue','ratio']};
const SECTORS={'Information Technology':'Information Technology','Health Care':'Health Care','Industrials':'Industrials','Consumer Discretionary':'Consumer Discretionary','Consumer Staples':'Consumer Staples','Energy':'Energy','Materials':'Materials','Utilities':'Utilities','Communication Services':'Communication Services','Real Estate':'Real Estate'};
const COLORS=['var(--series-1)','var(--series-2)','var(--series-3)','var(--ink-2)'];
const CORR=['operating_margin','net_margin','asset_turnover','equity_ratio','cash_ratio','rd_ratio'];
const SHORT=['Op.','Net','Turn.','Eq.','Cash','R&D'];
const PRIVACY_FEATURES=['revenue','asset_turnover','operating_margin','equity_ratio','cash_ratio'];
const score=x=>S.finite(x)?`${(100*x).toFixed(1)}%`:'N/A';
const distance=x=>S.finite(x)?x.toPrecision(4):'N/A';
let current=null, requestId=0, controller=null, timer=null;
const picks={};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=x=>S.finite(x)?x.toLocaleString('en-US',{maximumFractionDigits:2}):'Missing';
function fmt(value,key){if(!S.finite(value))return 'Missing';const type=METRICS[key]?.[1];if(type==='ratio')return `${(100*value).toFixed(1)}%`;if(type==='times')return `${value.toFixed(2)}×`;if(type==='money'){const a=Math.abs(value);return a>=1e9?`$${(value/1e9).toFixed(2)}B`:a>=1e6?`$${(value/1e6).toFixed(1)}M`:`$${num(value)}`;}return num(value);}
function options(el,selected){el.innerHTML=Object.entries(METRICS).map(([k,v])=>`<option value="${k}">${v[0]}</option>`).join('');el.value=selected;}
options($('metric'),'net_margin');options($('scatterX'),'asset_turnover');options($('scatterY'),'operating_margin');
CORR.forEach((a,i)=>CORR.slice(i+1).forEach(b=>{const o=document.createElement('option');o.value=`${a}|${b}`;o.textContent=`${METRICS[a][0]} × ${METRICS[b][0]}`;$('trendPair').append(o);}));
function groups(){return [...current.methods,{id:'real',label:'Real reports',rows:current.real}];}
function cards(left,right,draw){$(left).replaceChildren();$(right).replaceChildren();groups().forEach((g,i)=>{const card=document.createElement('div');card.className='card';card.style.setProperty('--series',COLORS[i]);$(i===3?right:left).append(card);draw(card,g,i);});}
function tooltip(event,text){const box=$('tooltip');box.textContent=text;box.hidden=false;box.style.left=`${Math.min(event.clientX+12,innerWidth-300)}px`;box.style.top=`${Math.max(5,Math.min(event.clientY+12,innerHeight-100))}px`;}
function hover(selection,text){selection.on('mousemove',(event,d)=>tooltip(event,text(d))).on('mouseleave',()=>{$('tooltip').hidden=true;});}
function svgFrame(card,w=270,h=190){return d3.select(card).append('svg').attr('viewBox',`0 0 ${w} ${h}`).attr('role','img');}
function paddedDomain(arr){let [lo,hi]=d3.extent(arr);if(lo===undefined)return [0,1];if(lo===hi){const p=Math.abs(lo)*.05||.05;return [lo-p,hi+p];}return [lo,hi];}
function drawHistograms(){const key=$('metric').value, all=groups().map(g=>S.values(g.rows,key)), useLog=$('logScale').checked&&METRICS[key][1]==='money';
 const transform=v=>useLog?Math.sign(v)*Math.log10(1+Math.abs(v)/1e6):v;
 const inverse=v=>useLog?Math.sign(v)*(10**Math.abs(v)-1)*1e6:v;
 const domain=paddedDomain(all.flat().map(transform)), x=d3.scaleLinear().domain(domain).range([39,257]);
 const thresholds=d3.range(1,19).map(i=>domain[0]+i*(domain[1]-domain[0])/19);
 const binner=d3.bin().domain(domain).thresholds(thresholds);
 const hist=all.map(a=>binner(a.map(transform)));
 const ymax=d3.max(hist,(bins,i)=>d3.max(bins,b=>b.length/(all[i].length||1)))||1;
 const y=d3.scaleLinear().domain([0,ymax*1.08]).range([149,12]);
 cards('syntheticHist','realHist',(card,g,i)=>{card.innerHTML=`<h3>${esc(g.label)} · ${METRICS[key][0]}</h3>`;
  if(!all[i].length){card.insertAdjacentHTML('beforeend','<div class="empty">No reported values available for this metric</div>');return;}
  const svg=svgFrame(card,270,180).attr('aria-label',`${g.label} ${METRICS[key][0]} distribution`);
  svg.append('g').attr('class','grid').attr('transform','translate(39,0)').call(d3.axisLeft(y).ticks(3).tickSize(-218).tickFormat('')).call(g=>g.select('.domain').remove());
  const bars=svg.append('g').selectAll('rect').data(hist[i]).join('rect').attr('x',b=>x(b.x0)+.8).attr('width',b=>Math.max(0,x(b.x1)-x(b.x0)-1.6)).attr('y',b=>y(b.length/all[i].length)).attr('height',b=>y(0)-y(b.length/all[i].length)).attr('fill',COLORS[i]).attr('opacity',.8);
  hover(bars,b=>`${fmt(inverse(b.x0),key)} – ${fmt(inverse(b.x1),key)}\n${b.length} samples · ${(100*b.length/all[i].length).toFixed(1)}%`);
  svg.append('g').attr('class','axis').attr('transform','translate(0,149)').call(d3.axisBottom(x).ticks(3).tickFormat(v=>fmt(inverse(v),key)));
  svg.append('g').attr('class','axis').attr('transform','translate(39,0)').call(d3.axisLeft(y).ticks(3).tickFormat(d3.format('.0%')));
  card.insertAdjacentHTML('beforeend',`<div class="stats"><span>Median <strong>${fmt(S.quantile(all[i],.5),key)}</strong></span><span>Valid: ${all[i].length}/${g.rows.length}</span></div><div class="metric-details">IQR ${fmt(S.quantile(all[i],.25),key)} – ${fmt(S.quantile(all[i],.75),key)}${i<3?`<br>KS distance ${num(S.ks(all[i],all[3]))} (0 = identical empirical distributions)`:''}</div>`);
 });
}
function drawScatter(){const a=$('scatterX').value,b=$('scatterY').value;
 const datasets=groups().map(g=>g.rows.filter(r=>S.finite(r.metrics[a])&&S.finite(r.metrics[b])));
 const x=d3.scaleLinear().domain(paddedDomain(datasets.flat().map(r=>r.metrics[a]))).range([44,255]);
 const y=d3.scaleLinear().domain(paddedDomain(datasets.flat().map(r=>r.metrics[b]))).range([148,13]);
 cards('syntheticScatter','realScatter',(card,g,i)=>{const data=datasets[i], c=S.correlation(g.rows.map(r=>r.metrics),a,b);card.innerHTML=`<h3>${METRICS[a][0]} × ${METRICS[b][0]}</h3>`;
  if(!data.length){card.insertAdjacentHTML('beforeend','<div class="empty">No valid paired observations</div>');return;}
  const svg=svgFrame(card,270,185).attr('aria-label',`${g.label} scatter plot`);
  svg.append('g').attr('class','axis').attr('transform','translate(0,148)').call(d3.axisBottom(x).ticks(3).tickFormat(v=>fmt(v,a)));
  svg.append('g').attr('class','axis').attr('transform','translate(44,0)').call(d3.axisLeft(y).ticks(3).tickFormat(v=>fmt(v,b)));
  const shown=data.filter((_,j)=>j%Math.max(1,Math.ceil(data.length/500))===0);
  const dots=svg.append('g').selectAll('circle').data(shown).join('circle').attr('cx',r=>x(r.metrics[a])).attr('cy',r=>y(r.metrics[b])).attr('r',2.6).attr('fill',COLORS[i]).attr('opacity',.57);
  hover(dots,r=>`${r.name} · ${SECTORS[r.sector]||r.sector}\n${METRICS[a][0]} ${fmt(r.metrics[a],a)}\n${METRICS[b][0]} ${fmt(r.metrics[b],b)}`);
  card.insertAdjacentHTML('beforeend',`<div class="stats"><span>Pearson r <strong>${c.r===null?'N/A':c.r.toFixed(3)}</strong></span><span>Valid pairs ${c.n} · Shown: ${shown.length}</span></div>`);
 });
}
function matrix(rows){return CORR.flatMap((a,i)=>CORR.map((b,j)=>({...S.correlation(rows.map(r=>r.metrics),a,b),i,j,a,b})));}
function drawCorrelations(){const reference=matrix(current.real), color=d3.scaleDiverging([-1,0,1],d3.interpolateRdBu);
 cards('syntheticCorr','realCorr',(card,g,index)=>{const cells=matrix(g.rows);card.innerHTML='<h3>Correlation matrix · Pearson r</h3>';
  const svg=svgFrame(card,270,218).attr('aria-label',`${g.label} Correlation matrix`), cell=29,x0=68,y0=14;
  const boxes=svg.selectAll('rect').data(cells).join('rect').attr('x',d=>x0+d.j*cell).attr('y',d=>y0+d.i*cell).attr('width',cell-1).attr('height',cell-1).attr('rx',2).attr('fill',d=>d.r===null?'var(--grid)':color(d.r));
  hover(boxes,d=>`${METRICS[d.a][0]} × ${METRICS[d.b][0]}\nr = ${d.r===null?'N/A':d.r.toFixed(3)} · Valid pairs ${d.n}`);
  svg.selectAll('.coefficient').data(cells).join('text').attr('class','coefficient').attr('x',d=>x0+d.j*cell+14).attr('y',d=>y0+d.i*cell+17).attr('text-anchor','middle').attr('fill',d=>d.r!==null&&Math.abs(d.r)>.6?'white':'var(--ink-1)').text(d=>d.r===null?'—':d.r.toFixed(1));
  SHORT.forEach((label,i)=>{svg.append('text').attr('x',x0-6).attr('y',y0+i*cell+18).attr('text-anchor','end').attr('fill','var(--ink-muted)').attr('font-size',10).text(label);svg.append('text').attr('x',x0+i*cell+14).attr('y',y0+6*cell+14).attr('text-anchor','middle').attr('fill','var(--ink-muted)').attr('font-size',9).text(label);});
  const differences=cells.filter((d,k)=>d.i<d.j&&d.r!==null&&reference[k].r!==null).map(d=>Math.abs(d.r-reference.find(x=>x.i===d.i&&x.j===d.j).r));
  card.insertAdjacentHTML('beforeend',`<div class="corr-key">−1 negative · 0 · +1 positive; — insufficient data or constant values</div>${index<3?`<div class="stats"><span>Mean absolute difference from real correlations <strong>${differences.length?d3.mean(differences).toFixed(3):'N/A'}</strong></span><span>${differences.length} pairs</span></div>`:''}`);
 });
}
const LABELS={revenue:'Revenue',revenues:'Revenue',cost_of_revenue:'Cost of revenue',gross_profit:'Gross profit',research_and_development:'Research and development',selling_general_and_administrative:'Selling, general and administrative',selling_general_administrative:'Selling, general and administrative',other_operating_expense:'Other operating expenses',other_operating_expenses:'Other operating expenses',operating_income:'Operating income',interest_expense:'Interest expense',pretax_income:'Pretax income',income_tax:'Income tax',income_tax_expense:'Income tax',net_income:'Net income',cash:'Cash',cash_and_equivalents:'Cash and equivalents',receivables:'Receivables',accounts_receivable:'Receivables',inventory:'Inventory',total_current_assets:'Total current assets',property_plant_and_equipment:'Property, plant and equipment, net',property_plant_equipment_net:'Property, plant and equipment, net',goodwill:'Goodwill',total_assets:'Total assets',total_current_liabilities:'Total current liabilities',long_term_debt:'Long-term debt',total_equity:'Total equity',total_stockholders_equity:'Total equity',total_liabilities:'Total liabilities',total_liabilities_and_equity:'Total liabilities and equity',net_cash_from_operating_activities:'Net cash from operating activities',cash_from_operations:'Net cash from operating activities',net_cash_from_investing_activities:'Net cash from investing activities',cash_from_investing:'Net cash from investing activities',net_cash_from_financing_activities:'Net cash from financing activities',cash_from_financing:'Net cash from financing activities',net_change_in_cash:'Net change in cash',cash_beginning_of_year:'Opening cash',cash_end_of_year:'Closing cash',depreciation_and_amortization:'Depreciation and amortization',capital_expenditures:'Capital expenditures',dividends_paid:'Dividends paid',share_repurchases:'Share repurchases',share_repurchases_net:'Net share repurchases',other_assets:'Other assets',other_current_assets:'Other current assets',other_noncurrent_assets:'Other non-current assets',other_liabilities:'Other liabilities',other_noncurrent_liabilities:'Other non-current liabilities',other_equity:'Other equity',changes_in_working_capital_and_other:'Working capital and other changes',other_operating_items:'Other operating items',other_investing_activities:'Other investing activities',other_financing_activities:'Other financing activities',net_change_in_long_term_debt:'Net change in long-term debt',assets:'Assets',liabilities:'Liabilities',equity:'Equity',operating_expenses:'Operating expenses',operating_activities:'Operating activities',investing_activities:'Investing activities',financing_activities:'Financing activities',total_liabilities:'Total liabilities'};
function tableHtml(object){let rows=[];function walk(obj,prefix=''){for(const [key,value] of Object.entries(obj||{})){const label=LABELS[key]||METRICS[key]?.[0]||key;if(value!==null&&typeof value==='object')walk(value,`${prefix}${label} / `);else rows.push(`<tr><td>${esc(prefix+label)}</td><td>${esc(typeof value==='string'?value:num(value))}</td></tr>`);}}walk(object);return `<div class="table-wrap"><table><tbody>${rows.join('')}</tbody></table></div>`;}
function realStatement(row,kind){const m=row.metrics;const keys=kind==='income_statement'?['revenue','cost_of_revenue','gross_profit','rd_expense','sga_expense','operating_income','pretax_income','income_tax','net_income']:kind==='balance_sheet'?['cash','receivables','inventory','current_assets','ppe_net','goodwill','total_assets','current_liabilities','long_term_debt','total_liabilities','stockholders_equity']:['net_income','depreciation_amortization','operating_cash_flow','capex','dividends_paid'];return Object.fromEntries(keys.map(k=>[k,m[k]??null]));}
function drawStatements(){cards('syntheticStatements','realStatements',(card,g,i)=>{const picker=document.createElement('select');picker.className='statement-picker';picker.setAttribute('aria-label',`${g.label} company`);for(const row of g.rows){const option=document.createElement('option');option.value=row.id;option.textContent=`${row.id} · ${row.name}`;picker.append(option);}picker.value=g.rows.some(r=>r.id===picks[g.id])?picks[g.id]:g.rows[0]?.id;card.append(picker);const content=document.createElement('div');card.append(content);function update(){picks[g.id]=picker.value;const row=g.rows.find(r=>r.id===picker.value);if(!row)return;const kind=$('statement').value;let object=i===3?realStatement(row,kind):row.statements[kind];content.innerHTML=`<div class="unit">FY ${row.fiscal_year} · Units: ${i===0||i===2?'USD millions':'USD'}${row.template?` · Template ${esc(row.template)}`:''}</div>${tableHtml(object)}${i===3?'<small>Real statement details use extracted metadata metrics. Unreported fields remain missing; this is not a complete set of disclosures.</small>':'<small>Accounting checks passed. Method 2 does not include cash/equity roll-forward checks.</small>'}`;}picker.addEventListener('change',update);update();});}
function drawDiagnostics(){cards('diagnostics','realDiagnostics',(card,g,i)=>{if(i===3){const missing=Object.keys(METRICS).map(k=>`${METRICS[k][0]}: ${S.values(g.rows,k).length}/${g.rows.length}`).join('<br>');card.innerHTML=`<h3>Real data coverage</h3><div class="diagnostic-row"><span>Fixed reference count</span><strong>${g.rows.length}</strong></div><small>${missing}</small><p class="metric-details">Real distributions retain reported values. Missing is not zero; real values are not adjusted to force reconciliation.</p>`;return;}
 const d=g.diagnostics;card.innerHTML=`<h3>${esc(g.label)} · Generation and reconciliation</h3><div class="diagnostic-row"><span>Passed native accounting checks</span><strong>${g.rows.filter(r=>r.checks.length===0).length}/${g.rows.length}</strong></div><div class="diagnostic-row"><span>Attempts / rejections</span><strong>${d.attempted} / ${d.rejected}</strong></div><div class="diagnostic-row"><span>Rejection rate</span><strong>${(100*d.rejected/d.attempted).toFixed(1)}%</strong></div><div class="diagnostic-row"><span>Accepted samples with adjustments</span><strong>${d.adjusted===null?'Not separately recorded':`${d.adjusted}/${d.accepted}`}</strong></div>${d.unique_templates!==undefined?`<div class="diagnostic-row"><span>Unique templates / reuse count</span><strong>${d.unique_templates} / ${d.template_reuses}</strong></div>`:''}<small>${esc(d.adjustment_note)}</small>`;
 if(d.mean_absolute_adjustment){card.insertAdjacentHTML('beforeend',`<details><summary>Adjustments by metric</summary><small>Absolute ratio changes; 0.01 means 1 percentage point, except for asset turnover. Includes rounding.</small>${tableHtml(d.mean_absolute_adjustment)}</details>`);}
 if(d.rejection_reasons&&Object.keys(d.rejection_reasons).length)card.insertAdjacentHTML('beforeend',`<details><summary>Rejection reasons (multiple per attempt possible)</summary>${tableHtml(d.rejection_reasons)}</details>`);
 });}
function drawSectorCoverage(){
 const entries=Object.entries(current.model_info).sort((a,b)=>b[1].n-a[1].n);
 const card=$('sectorCoverage');card.innerHTML='<h3>Real-report sector coverage behind this sample</h3>';
 if(!entries.length){card.insertAdjacentHTML('beforeend','<div class="empty">No sector data available</div>');return;}
 const w=760,barH=22,gap=8,left=190,top=10,right=50,h=entries.length*(barH+gap)+top;
 const svg=d3.select(card).append('svg').attr('viewBox',`0 0 ${w} ${h}`).attr('role','img').attr('aria-label','Real companies per sector backing this sample');
 const max=d3.max(entries,([,d])=>d.n)||1,x=d3.scaleLinear().domain([0,max]).range([0,w-left-right]);
 const bars=svg.append('g').selectAll('rect').data(entries).join('rect')
  .attr('x',left).attr('y',(_,i)=>top+i*(barH+gap)).attr('height',barH)
  .attr('width',([,d])=>Math.max(2,x(d.n))).attr('fill','var(--series-1)').attr('opacity',.85).attr('rx',3);
 hover(bars,([sector,d])=>`${SECTORS[sector]||sector}\n${d.n} real companies${d.fit_policy==='pooled_singleton'?' · pooled model (too few for its own fit)':` · shrinkage ${d.shrinkage.toFixed(2)}`}`);
 svg.append('g').selectAll('text').data(entries).join('text')
  .attr('x',left-10).attr('y',(_,i)=>top+i*(barH+gap)+barH/2+4).attr('text-anchor','end')
  .attr('font-size',11).attr('fill','var(--ink-2)').text(([sector])=>SECTORS[sector]||sector);
 svg.append('g').selectAll('text').data(entries).join('text')
  .attr('x',([,d])=>left+Math.max(2,x(d.n))+6).attr('y',(_,i)=>top+i*(barH+gap)+barH/2+4)
  .attr('font-size',11).attr('fill','var(--ink-2)').text(([,d])=>d.n);
 const total=entries.reduce((s,[,d])=>s+d.n,0);
 card.insertAdjacentHTML('beforeend',`<p class="metric-details">${entries.length} sector${entries.length===1?'':'s'} represented, ${total} real compan${total===1?'y':'ies'} total in the current scope. A sector modeled on very few companies (pooled fallback) is noted on hover. This is the one measure here of how much real-world situational variety backs the sample -- not scenario labels like Method 2's hand-authored ones, since Method 1 and Method 3 condition on sector rather than discrete scenarios.</p>`);
}
function sourcePanel(){
 const p=current.provenance,counts={};
 p.excluded.forEach(r=>counts[r.reason]=(counts[r.reason]||0)+1);
 const modelDetails=Object.entries(current.model_info).map(([s,d])=>
  `<p><strong>${esc(SECTORS[s]||s)}</strong>: ${d.n}  companies; ${d.fit_policy==='pooled_singleton'?'Insufficient sector samples; using the pooled joint model':`Correlation shrinkage weight toward the pooled estimate:  ${d.shrinkage.toFixed(2)}`}. Required columns using pooled-median fallback: ${esc(d.pooled_fallback.join(', ')||'None')}. Optional ratios omitted under Method 1’s sparse-sector policy (synthetic values are zero): ${esc(d.optional_zero.join(', ')||'None')}. </p><small>Original missing counts: ${esc(JSON.stringify(d.missing))}. Other missing cells use sector medians when fitting correlations. Non-fallback empirical marginals use observed values only.</small>`).join('');
 $('source').innerHTML=`<p><strong>Source: </strong>${esc(p.path)}</p>
  <p>Source: ${p.total} reports; eligible: ${p.eligible}; actual period-end years: ${p.period_years.join(', ')}; amounts are USD. The common population requires revenue, assets, equity, net income and a known non-financial sector.</p>
  <p>Excluded records: ${esc(JSON.stringify(counts))}. Charts additionally exclude missing values for each metric.</p>
  <ul>${current.limitations.map(x=>`<li>${esc(x)}</li>`).join('')}
   <li>Method 2 retains its original financial generation functions. The comparison tool assigns separate IDs to avoid the finite name-bank limit.</li>
   <li>Detailed facts match the exact report period end, preventing fiscal-label differences from mixing periods. Source fiscal-label warnings: ${esc(JSON.stringify(p.fiscal_label_warnings))}. Source files are unchanged.</li>
   <li>Asset turnover is revenue / year-end assets, matching the original method.</li>
   <li>KS distance is the maximum difference between two empirical CDFs. Correlation error is the mean absolute difference across valid off-diagonal pairs. Neither establishes statistical significance or a definitive ranking.</li>
  </ul><details><summary>Joint model: sector samples, missing values and fallbacks</summary>${modelDetails}</details>
  <details><summary>All excluded records</summary>${tableHtml(Object.fromEntries(p.excluded.map(r=>[r.ticker,r.reason])))}</details>`;
}
function drawPairTrends(){
 const [a,b]=$('trendPair').value.split('|');
 const complete=rows=>rows.filter(r=>S.finite(r.metrics[a])&&S.finite(r.metrics[b]));
 const datasets=groups().map(g=>complete(g.rows));
 const x=d3.scaleLinear().domain(paddedDomain(datasets.flat().map(r=>r.metrics[a]))).range([48,282]);
 const y=d3.scaleLinear().domain(paddedDomain(datasets.flat().map(r=>r.metrics[b]))).range([190,44]);
 const reference=current.real,realColor='var(--ink-2)';
 cards('syntheticTrends','realTrends',(card,g,i)=>{
  if(i===3){const fit=FinancialEvaluation.pairTrend(reference,reference,a,b).real;card.innerHTML=`<h3>Real reference · selected pair</h3><p>${METRICS[a][0]} × ${METRICS[b][0]}</p><div class="diagnostic-row"><span>Pearson r</span><strong>${fit.r===null?'N/A':fit.r.toFixed(3)}</strong></div><div class="diagnostic-row"><span>Complete observations</span><strong>${fit.n}/${reference.length}</strong></div><small>Dark circles and dashed lines in each comparison show this same real population. Colored diamonds and solid lines show the corresponding synthetic method. Missing values are excluded. A line is unavailable when X is constant or fewer than two complete observations exist.</small>`;return;}
  const pair=FinancialEvaluation.pairTrend(reference,g.rows,a,b);
  card.innerHTML=`<h3>${esc(g.label)} · Real vs. synthetic</h3><div class="stats"><span>Pair correlation similarity <strong>${score(pair.score)}</strong></span></div>`;
  const svg=svgFrame(card,300,246).attr('aria-label',`${g.label}: real versus synthetic ${METRICS[a][0]} and ${METRICS[b][0]}, correlation similarity ${score(pair.score)}`);
  const clipId=`trend-clip-${g.id}`;
  svg.append('defs').append('clipPath').attr('id',clipId).append('rect').attr('x',48).attr('y',44).attr('width',234).attr('height',146);
  svg.append('g').attr('class','grid').attr('transform','translate(48,0)').call(d3.axisLeft(y).ticks(4).tickSize(-234).tickFormat('')).call(g=>g.select('.domain').remove());
  const plot=svg.append('g').attr('clip-path',`url(#${clipId})`);
  [[datasets[3],pair.real,realColor,true],[datasets[i],pair.synthetic,COLORS[i],false]].forEach(([rows,fit,color,isReal])=>{
   const shown=rows.filter((_,j)=>j%Math.max(1,Math.ceil(rows.length/500))===0);
   const dots=plot.append('g').selectAll('path').data(shown).join('path').attr('d',d3.symbol().type(isReal?d3.symbolCircle:d3.symbolDiamond).size(isReal?13:18)).attr('transform',r=>`translate(${x(r.metrics[a])},${y(r.metrics[b])})`).attr('fill',color).attr('opacity',.48);
   hover(dots,r=>`${isReal?'Real':g.label} · ${r.name}\n${METRICS[a][0]} ${fmt(r.metrics[a],a)}\n${METRICS[b][0]} ${fmt(r.metrics[b],b)}`);
   if(fit.slope!==null){plot.append('path').attr('d',d3.line()(fit.extent.map(v=>[x(v),y(fit.intercept+fit.slope*v)]))).attr('fill','none').attr('stroke',color).attr('stroke-width',2).attr('stroke-dasharray',isReal?'5,3':null);}
  });
  svg.append('g').attr('class','axis').attr('transform','translate(0,190)').call(d3.axisBottom(x).ticks(4).tickFormat(v=>fmt(v,a)));
  svg.append('g').attr('class','axis').attr('transform','translate(48,0)').call(d3.axisLeft(y).ticks(4).tickFormat(v=>fmt(v,b)));
  svg.append('text').attr('x',165).attr('y',231).attr('text-anchor','middle').attr('font-size',10).attr('fill','var(--ink-muted)').text(METRICS[a][0]);
  svg.append('text').attr('transform','translate(11,117) rotate(-90)').attr('text-anchor','middle').attr('font-size',10).attr('fill','var(--ink-muted)').text(METRICS[b][0]);
  [[realColor,'Real',true],[COLORS[i],'Synthetic',false]].forEach(([color,label,isReal],j)=>{
   const legend=svg.append('g').attr('transform',`translate(${53+j*112},20)`);
   legend.append('line').attr('x1',0).attr('x2',20).attr('stroke',color).attr('stroke-width',2).attr('stroke-dasharray',isReal?'5,3':null);
   legend.append('path').attr('d',d3.symbol().type(isReal?d3.symbolCircle:d3.symbolDiamond).size(20)).attr('transform','translate(10,0)').attr('fill',color);
   legend.append('text').attr('x',26).attr('y',3).attr('font-size',10).attr('fill',color).text(label);
  });
  card.insertAdjacentHTML('beforeend',`<div class="metric-details">Pearson r · real ${pair.real.r===null?'N/A':pair.real.r.toFixed(3)} / synthetic ${pair.synthetic.r===null?'N/A':pair.synthetic.r.toFixed(3)}<br>Complete observations · real ${pair.real.n}/${reference.length} / synthetic ${pair.synthetic.n}/${g.rows.length}</div>`);
 });
}
function drawEvaluation(){
 const quality=current.methods.map(m=>{
  const columns=FinancialEvaluation.quality(current.real,m.rows,Object.keys(METRICS));
  const pairs=FinancialEvaluation.quality(current.real,m.rows,CORR);
  return {columnShapes:columns.columnShapes,columns:columns.columns,pairTrends:pairs.pairTrends,pairs:pairs.pairs};
 });
 const privacy=FinancialEvaluation.privacy(current.real,current.methods.map(m=>m.rows),PRIVACY_FEATURES,current.seed);
 // Keep the exported JSON auditable with feature sets, coverage, distances and settings.
 current.evaluation={implementation:'JavaScript; documented SDMetrics formulas on stated numeric subsets',scope:'In-sample; current sector and selected synthetic samples',qualityFeatures:Object.keys(METRICS),pairFeatures:CORR,privacyFeatures:PRIVACY_FEATURES,baselineSeed:current.seed,baselineMissingPolicy:'Uniform observed numeric ranges; no missing values',overfittingProtection:null,privacy,methods:current.methods.map((m,i)=>({id:m.id,...quality[i]}))};
 cards('syntheticQuality','realQuality',(card,g,i)=>{
  if(i===3){card.innerHTML='<h3>Real reference · evaluation scope</h3><p>All scores compare the selected synthetic sample to these fixed real reports.</p><p>Column coverage varies with missing disclosures; pair counts use complete observations.</p><small>Select a sector above to inspect within-sector fidelity. Pooled correlations can reflect differences between sectors. All comparisons are in-sample; small real sectors can produce unstable scores.</small>';return;}
  const q=quality[i],validCols=q.columns.filter(x=>x.score!==null).length,validPairs=q.pairs.filter(x=>x.score!==null).length;
  card.innerHTML=`<h3>${esc(g.label)}</h3><div class="diagnostic-row"><span>Mean KSComplement</span><strong>${score(q.columnShapes)}</strong></div><small>${validCols}/${q.columns.length} comparable columns</small><div class="diagnostic-row"><span>Column Pair Trends</span><strong>${score(q.pairTrends)}</strong></div><small>${validPairs}/${q.pairs.length} comparable pairs · Pearson</small><details><summary>KSComplement by column</summary><div class="table-wrap"><table><thead><tr><th>Column</th><th>Score</th><th>Valid real / synth.</th></tr></thead><tbody>${q.columns.map(x=>`<tr><td>${METRICS[x.key][0]}</td><td>${score(x.score)}</td><td>${x.realN} / ${x.syntheticN}</td></tr>`).join('')}</tbody></table></div></details><details><summary>Column Pair Trends · all pairs</summary><div class="table-wrap"><table><thead><tr><th>Pair</th><th>Score</th><th>r real / synth.</th><th>n real / synth.</th></tr></thead><tbody>${q.pairs.map(x=>`<tr><td>${METRICS[x.a][0]} × ${METRICS[x.b][0]}</td><td>${score(x.score)}</td><td>${x.realR===null?'N/A':x.realR.toFixed(3)} / ${x.syntheticR===null?'N/A':x.syntheticR.toFixed(3)}</td><td>${x.realN} / ${x.syntheticN}</td></tr>`).join('')}</tbody></table></div></details>`;
 });
 const all=[...privacy.methods,privacy.reference],domain=paddedDomain(all.flatMap(x=>x.distances));
 const x=d3.scaleLinear().domain(domain).range([39,257]),bin=d3.bin().domain(domain).thresholds(d3.range(1,19).map(i=>domain[0]+i*(domain[1]-domain[0])/19));
 const bins=all.map(d=>bin(d.distances)),y=d3.scaleLinear().domain([0,(d3.max(bins,(bs,i)=>d3.max(bs,b=>b.length/(all[i].distances.length||1)))||1)*1.08]).range([149,12]);
 cards('syntheticPrivacy','realPrivacy',(card,g,i)=>{
  const d=all[i];card.innerHTML=`<h3>${i===3?'Real → other real (self excluded)':`${esc(g.label)} → real`}</h3>`;
  if(d.distances.length){
   const svg=svgFrame(card,270,180).attr('aria-label',`${g.label} distance to closest real record`);
   const bars=svg.append('g').selectAll('rect').data(bins[i]).join('rect').attr('x',b=>x(b.x0)+.8).attr('width',b=>Math.max(0,x(b.x1)-x(b.x0)-1.6)).attr('y',b=>y(b.length/d.distances.length)).attr('height',b=>y(0)-y(b.length/d.distances.length)).attr('fill',COLORS[i]).attr('opacity',.8);
   hover(bars,b=>`DCR ${distance(b.x0)} – ${distance(b.x1)}\n${b.length} records · ${score(b.length/d.distances.length)}`);
   svg.append('g').attr('class','axis').attr('transform','translate(0,149)').call(d3.axisBottom(x).ticks(3).tickFormat(d3.format('.2g')));
   svg.append('g').attr('class','axis').attr('transform','translate(39,0)').call(d3.axisLeft(y).ticks(3).tickFormat(d3.format('.0%')));
  }else card.insertAdjacentHTML('beforeend','<p class="empty">DCR unavailable: no varying reference features or too few other real records.</p>');
  card.insertAdjacentHTML('beforeend',`<div class="diagnostic-row"><span>Median DCR</span><strong>${distance(d.median)}</strong></div><div class="diagnostic-row"><span>5th percentile / minimum</span><strong>${distance(d.p05)} / ${distance(d.minimum)}</strong></div><div class="diagnostic-row"><span>Zero-distance feature matches</span><strong>${score(d.zeroFraction)}</strong></div>${i<3?`<div class="diagnostic-row"><span>DCR baseline protection</span><strong>${score(d.score)}</strong></div><small>Uniform baseline median: ${distance(d.baselineMedian)} · ${g.rows.length} random rows</small>`:'<small>Leave-one-out distances use the same feature scaling. This is not a held-out privacy test.</small>'}<p class="metric-details">Active features: ${privacy.features.length}/${PRIVACY_FEATURES.length}${privacy.excluded.length?` · Excluded: ${privacy.excluded.map(k=>METRICS[k][0]).join(', ')}`:''}. Missingness contributes to distance. Higher DCR can also reflect unrealistic outliers.</p>`);
 });
}
function render(){const n=current.n;$('syntheticCount').textContent=`${n.toLocaleString()} per method`;$('realCount').textContent=`${current.real.length} reports`;$('scopeLabel').textContent=`${current.sector==='all'?'All mapped non-financial sectors':SECTORS[current.sector]||current.sector} · Real reference: ${current.real.length} fixed reports · Period-end years: ${current.provenance.period_years.join(' / ')}`;
 $('correlationScope').textContent=current.sector==='all'?' Sectors are pooled here; select a sector to inspect within-sector relationships.':' Within-sector relationships are shown.';
 $('methodHeadings').innerHTML=current.methods.map((m,i)=>`<div class="method-card" style="--series:${COLORS[i]}"><h3>${m.label}</h3><strong>${m.rows.length.toLocaleString()}</strong><span> companies</span><p>${['Independent sector quantiles + original accounting engine','Scale and perturb a real financial profile','Empirical marginals + regularized joint dependence'][i]}</p></div>`).join('');
 $('realHeading').innerHTML=`<div class="method-card" style="--series:${COLORS[3]}"><h3>SEC 10-K · Real reference</h3><strong>${current.real.length}</strong><span> reports</span><p>Fitting and comparison use the same real observations</p></div>`;
 drawHistograms();drawScatter();drawCorrelations();drawPairTrends();drawEvaluation();drawSectorCoverage();drawStatements();drawDiagnostics();sourcePanel();}
function workspaces(){return document.querySelectorAll('.workspace');}
function busy(value){document.body.classList.toggle('pending',value);workspaces().forEach(w=>w.setAttribute('aria-busy',String(value)));$('downloadCsv').disabled=value||!current;$('downloadJson').disabled=value||!current;}
async function generate(){const id=++requestId;if(controller)controller.abort();controller=new AbortController();busy(true);$('error').hidden=true;$('status').textContent='Generating and checking three sets of statements…';const n=Number($('sampleCount').value),seed=Number($('seed').value),sector=$('sector').value;const start=performance.now();try{const response=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({n,seed,sector}),signal:controller.signal});const result=await response.json();if(!response.ok)throw new Error(result.error||'Generation failed');if(id!==requestId)return;current=result;render();busy(false);$('status').textContent=`Updated · ${((performance.now()-start)/1000).toFixed(1)} s · seed ${seed}`;}catch(error){if(error.name==='AbortError'||id!==requestId)return;current=null;busy(false);$('error').hidden=false;$('error').textContent=`${error.message}. Adjust the settings and generate again.`;$('status').textContent='Generation incomplete';workspaces().forEach(w=>{w.setAttribute('aria-busy','false');w.style.opacity='.35';});}finally{if(current)workspaces().forEach(w=>{w.style.opacity='';});}}
function schedule(){clearTimeout(timer);requestId++;if(controller)controller.abort();busy(true);$('status').textContent='Settings changed; generation pending…';timer=setTimeout(generate,450);}
$('sampleCount').addEventListener('input',()=>{$('countLabel').textContent=Number($('sampleCount').value).toLocaleString();schedule();});
$('sector').addEventListener('change',schedule);$('seed').addEventListener('change',schedule);$('regenerate').addEventListener('click',()=>{clearTimeout(timer);generate();});
$('metric').addEventListener('change',()=>{if(current)drawHistograms();});$('logScale').addEventListener('change',()=>{if(current)drawHistograms();});
for(const id of ['scatterX','scatterY'])$(id).addEventListener('change',()=>{if(current)drawScatter();});$('statement').addEventListener('change',()=>{if(current)drawStatements();});
$('trendPair').addEventListener('change',()=>{if(current)drawPairTrends();});
$('sourceToggle').addEventListener('click',()=>{const open=$('source').hidden;$('source').hidden=!open;$('sourceToggle').setAttribute('aria-expanded',String(open));});
function download(text,type,name){const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);}
$('downloadJson').addEventListener('click',()=>{if(current)download(JSON.stringify(current,null,2),'application/json',`financial-comparison-${current.n}-seed${current.seed}.json`);});
$('downloadCsv').addEventListener('click',()=>{if(!current)return;const columns=['method','id','sector','fiscal_year',...Object.keys(METRICS)];const quote=v=>`"${String(v??'').replace(/"/g,'""')}"`;const rows=groups().flatMap(g=>g.rows.map(r=>[g.id,r.id,r.sector,r.fiscal_year,...Object.keys(METRICS).map(k=>r.metrics[k])]));download('\uFEFF'+[columns,...rows].map(r=>r.map(quote).join(',')).join('\r\n'),'text/csv;charset=utf-8',`financial-metrics-${current.n}-seed${current.seed}.csv`);});
(async()=>{try{const response=await fetch('/api/info');if(!response.ok)throw new Error('Unable to load source data');const info=await response.json();info.sectors.forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=SECTORS[s]||s;$('sector').append(o);});await generate();}catch(error){$('error').hidden=false;$('error').textContent=error.message;$('status').textContent='Loading failed. Check the local server.';busy(false);}})();
