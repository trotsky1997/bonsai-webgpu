const variant=new URLSearchParams(location.search).get('variant')||'optimized';
const workerPath=variant==='baseline'?'/benchmarks/baseline/worker.mjs':variant==='v020'?'/benchmarks/v020/worker.mjs':'/web/worker.mjs';
const worker=new Worker(workerPath,{type:'module'}),listeners=new Map();let serial=1000,info;
const status=document.querySelector('#status'),log=document.querySelector('#log');
const append=s=>{log.textContent=(log.textContent+'\n'+s).slice(-12000);};
let resolveReady,rejectReady;const ready=new Promise((r,j)=>{resolveReady=r;rejectReady=j;});
worker.onmessage=({data})=>{
 if(data.type==='loading')status.textContent=data.message;
 else if(data.type==='ready'){info=data;status.textContent='模型已加载，等待验证任务。';append(JSON.stringify(data,null,2));resolveReady();}
 else if(data.type==='error'&&!data.id){status.textContent=data.message;rejectReady(Error(data.message));}
 else listeners.get(data.id)?.(data);
};
worker.onerror=e=>{status.textContent=e.message;rejectReady(e);};
const benchParams=new URLSearchParams(location.search);
worker.postMessage({op:'init',context:Number(benchParams.get('context')??2048),kv_bits:Number(benchParams.get('kv_bits')??32),batch:8,fast:variant!=='reference'});
function command(op,args={}){
 const id=++serial;let content='',first=null,tool_calls=[];const start=performance.now();
 return new Promise((resolve,reject)=>{listeners.set(id,event=>{
  if(event.type==='error'){listeners.delete(id);reject(Error(event.message));}
  if(event.type==='tool_calls')tool_calls=event.tool_calls;
  if(event.type==='delta'){content+=event.text;if(first===null)first=performance.now()-start;}
  if(['done','reset','diagnosed','info','traced','checked'].includes(event.type)){listeners.delete(id);resolve({...event,content,tool_calls,wall_ttft_ms:first,wall_ms:performance.now()-start});}
 });worker.postMessage({op,id,...args});});
}
async function run(job){
 status.textContent='正在验证：'+job.job_id;
 let result;
 if(job.op==='load-check'){
  const rows=[];
  for(let round=0;round<(job.rounds??5);round++){
   const probe=new Worker(workerPath,{type:'module'});
   const cache_name=job.cold?'bonsai-load-test-'+job.job_id+'-'+round:undefined;
   try{rows.push(await new Promise((resolve,reject)=>{probe.onmessage=({data})=>{if(data.type==='ready')resolve(data);else if(data.type==='error')reject(Error(data.message));};probe.onerror=reject;probe.postMessage({op:'init',context:2048,batch:8,cache_name,force_network:!!job.cold,prefetch:job.prefetch});}));}
   finally{probe.terminate();if(cache_name)await caches.delete(cache_name);}
  }
  result={result:rows};
 }
 else if(job.op==='generate'){await command('reset');result=await command('generate',{body:job.body});}
 else if(job.op==='diagnose')result=await command('diagnose',{job:job.job});
 else if(job.op==='info')result=await command('info');
 else if(job.op==='kv-mode')result=await command('kv-mode',{bits:job.bits});
 else if(job.op==='kv-check')result=await command('kv-check',{count:job.count});
 else if(job.op==='tiled-mode')result=await command('tiled-mode',{enabled:job.enabled});
 else if(job.op==='profile')result=await command('profile',{token:job.token,position:job.position});
 else if(job.op==='trace')result=await command('trace',{token:job.token});
 else if(job.op==='kernel-check')result=await command('kernel-check');
 else if(job.op==='equivalence-check')result=await command('equivalence-check',{text:job.text,limit:job.limit});
 else if(job.op==='ngram-check')result=await command('ngram-check',{text:job.text});
 else if(job.op==='candidate-check')result=await command('candidate-check');
 else if(job.op==='tokenize-check')result=await command('tokenize-check',{texts:job.texts});
 else if(job.op==='batch-check')result=await command('batch-check',{text:job.text,limit:job.limit});
 else throw Error('Unknown benchmark op');
 const out={job_id:job.job_id,...result};
 append(JSON.stringify({...out,result:out.result?.tokens?{mean_nll:out.result.mean_nll,tokens:out.result.tokens.length}:undefined}));
 await fetch('/benchmark-result',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(out)});
 status.textContent='已完成：'+job.job_id;
}
await ready;
for(;;){
 let currentJob;
 try{const job=await (await fetch('/benchmark-next?variant='+variant,{cache:'no-store'})).json();if(job){currentJob=job;await run(job);}else await new Promise(r=>setTimeout(r,300));}
 catch(e){append('ERROR: '+e.message);await fetch('/benchmark-result',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:currentJob?.job_id??'load-error',error:e.message})});await new Promise(r=>setTimeout(r,1000));}
}
