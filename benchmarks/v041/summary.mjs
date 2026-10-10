import {readFile} from 'node:fs/promises';
const read=async name=>JSON.parse(await readFile(new URL(name,import.meta.url),'utf8'));
const median=xs=>{const sorted=[...xs].sort((a,b)=>a-b);return sorted[Math.floor(sorted.length/2)];};
const speeds=rows=>{const xs=rows.map(r=>(r.completion_tokens-1)*1000/r.decode_ms);return {median:median(xs),min:Math.min(...xs),max:Math.max(...xs)};};
const kv=await read('kv-benchmark.json');
for(const bits of [32,4])console.log('KV',bits,speeds(Array.from({length:7},(_,i)=>kv[`kv-speed-${i}-${bits}`])));
for(const bits of [32,4])console.log('NLL',bits,kv[`quality${bits}-nll`].result.mean_nll);
const ngram=await read('ngram-benchmark.json');
for(const workload of ['prose','copy']){
 const rows=ngram.runs.filter(r=>r.workload===workload);
 const exact=Array.from({length:8},(_,i)=>i-1).every(round=>JSON.stringify(rows.find(r=>r.round===round&&!r.enabled)?.tokens)===JSON.stringify(rows.find(r=>r.round===round&&r.enabled)?.tokens));
 console.log('ngram',workload,'paired output exact:',exact);
 for(const enabled of [false,true])console.log(enabled,speeds(rows.filter(r=>r.round>=0&&r.enabled===enabled)));
}
