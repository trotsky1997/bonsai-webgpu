import init, {Engine} from './pkg/bonsai_webgpu.js';
let cacheName,forceNetwork=false;
let engine, ready, busy=false, cancelled=false, cached=[], loadInfo;
const emit=value=>postMessage(value);
let loadStats=[],cachePromise;
async function asset(path,expected) {
 const started=performance.now();
 const url=new URL('../model/'+path,import.meta.url);if(expected)url.searchParams.set('sha256',expected);
 const cache=await(cachePromise??=caches.open(cacheName||'bonsai-webgpu-verified-v2'));
 let response=await cache.match(url);const cached=!!response,matched=performance.now();
 if(!response){response=await fetch(url,{cache:forceNetwork?'reload':'default'});if(!response.ok)throw Error('下载失败: '+path);}
 const bytes=new Uint8Array(await response.arrayBuffer());const read=performance.now();
 if(expected&&!cached){
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
  if(hash!==expected)throw Error('权重校验失败，请重试: '+path);
 }
 // Only verified downloads enter this content-addressed cache. A cache hit
 // has already passed SHA-256 and cannot be confused with another revision.
 if(!cached)await cache.put(url,new Response(bytes,{headers:{'Content-Type':'application/octet-stream'}}));
 loadStats.push({path,cached,cache_ms:matched-started,read_ms:read-matched,verify_cache_ms:performance.now()-read});return bytes;
}
async function load(context=2048,batch=8) {
 const started=performance.now();emit({type:'loading',message:'正在初始化 WebGPU…',percent:0});
 if(!navigator.gpu)throw Error('当前浏览器没有 WebGPU；请使用支持 WebGPU 的新版 Chrome、Edge 或 Safari。');
 const wasmPromise=init();
 const manifestPromise=fetch(new URL('../model/manifest.json',import.meta.url),{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('模型清单下载失败');return r.json();});
 const manifest=await manifestPromise;
 const tk=manifest.tokenizer??{name:'tokenizer.json'};
 const tokenizerPromise=asset(tk.name,tk.sha256);
 // Three bounded uploads in flight hide cache I/O behind tokenizer and GPU setup.
 const prefetched=new Map();const prefetch=i=>{if(i<manifest.files.length&&!prefetched.has(i)){const f=manifest.files[i];prefetched.set(i,asset(f.name,f.sha256));}};
 for(let i=0;i<3;i++)prefetch(i);
 await wasmPromise;const wasm=performance.now();const tokenizer=await tokenizerPromise;const tok=performance.now();
 engine=await Engine.create(tokenizer,context,batch);const created=performance.now();let upload=0;
 let loaded=0,total=manifest.files.reduce((s,f)=>s+f.bytes,0);
 for(let i=0;i<manifest.files.length;i++){
  const file=manifest.files[i];const bytes=await prefetched.get(i);prefetched.delete(i);prefetch(i+3);
  const t=performance.now();engine.load_blob(bytes,i-1);upload+=performance.now()-t;loaded+=file.bytes;
  emit({type:'loading',message:`正在加载模型 ${Math.round(loaded/total*100)}%`,percent:loaded/total*100});
 }
 loadInfo={runtime_version:'0.2.0',backend:'browser-webgpu',kernel:'signmag-exact-centered-g32',weight_bits:1,activation_bits:4,context,batch,adapter:engine.adapter(),load_ms:performance.now()-started,load_stages:{wasm_ms:wasm-started,tokenizer_ms:tok-wasm,engine_ms:created-tok,upload_ms:upload,files:loadStats},memory:engine.memory(),checkpoint_sha256:manifest.checkpoint_sha256};
 emit({type:'ready',...loadInfo});return loadInfo;
}
export function renderPrompt(body){
 if(!Array.isArray(body.messages)||!body.messages.length)throw Error('messages required');
 let prompt='';const lastUser=body.messages.findLastIndex(m=>m.role==='user');
 for(const [index,message] of body.messages.entries()){
  const role=message.role==='developer'?'system':message.role;if(!['system','user','assistant'].includes(role)||message.tool_calls?.length)throw Error('只支持文本对话');
  let content=Array.isArray(message.content)?message.content.map(p=>{if(p.type!=='text')throw Error('只支持文本消息');return p.text;}).join(''):message.content||'';
  let reasoning=message.reasoning_content||'';
  if(role==='assistant'&&content.includes('</think>')){if(!reasoning)reasoning=content.split('</think>')[0].split('<think>').at(-1).trim();content=content.split('</think>').at(-1).replace(/^\n+/,'');}
  prompt+=`<|im_start|>${role}\n`;
  if(role==='assistant'&&index>lastUser&&(index===body.messages.length-1||reasoning))prompt+=`<think>\n${reasoning.trim()}\n</think>\n\n`;
  prompt+=content+'<|im_end|>\n';
 }
 if(body.continue_final_message){if(body.messages.at(-1).role!=='assistant')throw Error('Continuation needs assistant');return prompt.slice(0,-'<|im_end|>\n'.length);}
 return prompt+'<|im_start|>assistant\n<think>\n\n</think>\n\n';
}
function sample(candidates,body,rng=Math.random){
 if((body.temperature??0.5)<=0)return candidates[0][0];
 const max=candidates[0][1],temperature=body.temperature??0.5;
 let weights=candidates.map(([,s])=>Math.exp((s-max)/temperature)),sum=weights.reduce((a,b)=>a+b,0),cumulative=0,keep=weights.length;
 for(let i=0;i<weights.length;i++){cumulative+=weights[i]/sum;if(cumulative>=(body.top_p??0.85)){keep=i+1;break;}}
 weights=weights.slice(0,keep);let choice=rng()*weights.reduce((a,b)=>a+b,0);
 for(let i=0;i<keep;i++){choice-=weights[i];if(choice<=0)return candidates[i][0];}return candidates[keep-1][0];
}
async function generate(id,body){
 if(busy)throw Error('模型正在回答另一个请求');busy=true;cancelled=false;
 try{
  await ready;const started=performance.now();const prompt=body.prompt??renderPrompt(body);const ids=Array.from(engine.encode(prompt));
  if(!ids.length||ids.length>=loadInfo.context)throw Error(`上下文超过 ${loadInfo.context} token，请新建对话或缩短内容。`);
  const requestedMax=body.max_completion_tokens??body.max_tokens??body.n_predict??512;
  const max=Math.min(requestedMax===-1?loadInfo.context:requestedMax,loadInfo.context-ids.length);
  if(!Number.isInteger(max)||max<0)throw Error('Invalid max_tokens');
  const requested=body.top_k??20;if(!Number.isInteger(requested)||requested<1||requested>64)throw Error('浏览器版本 top_k 支持 1–64');
  if((body.repeat_penalty??1)!==1||(body.presence_penalty??0)!==0||(body.frequency_penalty??0)!==0)throw Error('浏览器版本当前要求 repeat_penalty=1、presence_penalty=0、frequency_penalty=0');
  const topk=(body.temperature??0.5)<=0?1:requested;
  let prefix=0;while(prefix<Math.min(cached.length,ids.length-1)&&cached[prefix]===ids[prefix])prefix++;
  engine.truncate(prefix);cached.length=prefix;let candidates;
  emit({type:'progress',id,prompt_tokens:ids.length,cached_tokens:prefix,processed_tokens:prefix});
  for(let i=prefix;i<ids.length;i+=loadInfo.batch){if(cancelled)break;const chunk=ids.slice(i,i+loadInfo.batch);candidates=await engine.forward(new Uint32Array(chunk),topk);cached.push(...chunk);emit({type:'progress',id,prompt_tokens:ids.length,cached_tokens:prefix,processed_tokens:i+chunk.length});}
  const prefill=performance.now()-started;const decodeStart=performance.now();let first=null,generated=[],sent='',finish='length';
  const stops=typeof body.stop==='string'?[body.stop]:body.stop??[];let stopped=false;
  let random=Math.random;
  if(Number.isInteger(body.seed)&&body.seed>=0){let state=(body.seed>>>0)||1;random=()=>{state^=state<<13;state^=state>>>17;state^=state<<5;return(state>>>0)/4294967296;};}
  for(let step=0;!cancelled&&step<max;step++){
   const token=sample(candidates,body,random);
   if(!body.ignore_eos&&(token===151645||token===151643)){finish='stop';break;}
   generated.push(token);let decoded=engine.decode(new Uint32Array(generated));
   let stopIndex=Infinity;for(const stop of stops){const at=decoded.indexOf(stop);if(at>=0)stopIndex=Math.min(stopIndex,at);}
   if(stopIndex!==Infinity){decoded=decoded.slice(0,stopIndex);stopped=true;finish='stop';}
   // Hold incomplete UTF-8 bytes and potential stop prefixes between tokens.
   let safe=decoded.endsWith('\ufffd')?decoded.slice(0,decoded.lastIndexOf('\ufffd')):decoded;
   if(!stopped&&stops.length){let hold=0;for(const stop of stops)for(let n=1;n<stop.length;n++)if(safe.endsWith(stop.slice(0,n)))hold=Math.max(hold,n);safe=safe.slice(0,safe.length-hold);}
   if(safe.length>sent.length){const text=safe.slice(sent.length);sent=safe;if(first===null)first=performance.now()-started;emit({type:'delta',id,text,token});}
   if(stopped)break;
   if(step+1<max){candidates=await engine.forward(new Uint32Array([token]),topk);cached.push(token);}
  }
  if(cancelled)finish='stop';
  if(!stopped){const decoded=engine.decode(new Uint32Array(generated));if(decoded.startsWith(sent)&&decoded.length>sent.length)emit({type:'delta',id,text:decoded.slice(sent.length)});}
  emit({type:'done',id,finish_reason:finish,prompt_tokens:ids.length,cached_tokens:prefix,completion_tokens:generated.length,prefill_ms:prefill,decode_ms:performance.now()-decodeStart,ttft_ms:first,tokens:generated,memory:engine.memory()});
 }finally{busy=false;}
}
async function diagnose(id,job){
 if(busy)throw Error('busy');busy=true;
 try{await ready;engine.reset();cached=[];const ids=job.tokens??Array.from(engine.encode(job.text));const nll=[],predictions=[],target_logits=[];
 for(let i=0;i<ids.length-1;i++){
  const logits=await engine.logits(ids[i]);let max=-Infinity,arg=0;
  for(let t=0;t<logits.length;t++)if(logits[t]>=max){max=logits[t];arg=t;}
  let sum=0;for(const value of logits)sum+=Math.exp(value-max);
  nll.push(Math.log(sum)+max-logits[ids[i+1]]);predictions.push(arg);target_logits.push(logits[ids[i+1]]);
  if(i%16===0)emit({type:'diagnostic-progress',id,processed:i+1,total:ids.length-1});
 }
 return {tokens:ids,nll,predictions,target_logits,mean_nll:nll.reduce((a,b)=>a+b,0)/nll.length,memory:engine.memory()};
 }finally{busy=false;}
}
onmessage=async({data})=>{
 const {op,id}=data;
 try{
  if(op==='init'){cacheName=data.cache_name;forceNetwork=!!data.force_network;ready=load(data.context??2048,data.batch??8);await ready;}
  else if(op==='cancel'){cancelled=true;}
  else if(op==='generate'){await generate(id,data.body);}
  else if(op==='reset'){await ready;if(busy)throw Error('busy');engine.reset();cached=[];emit({type:'reset',id});}
  else if(op==='diagnose'){const result=await diagnose(id,data.job);emit({type:'diagnosed',id,result});}
  else if(op==='info'){await ready;emit({type:'info',id,...loadInfo,memory:engine.memory()});}
  else if(op==='profile'){await ready;if(busy)throw Error('busy');emit({type:'checked',id,result:await engine.profile(data.token??9707,data.position??0)});}
  else if(op==='trace'){await ready;if(busy)throw Error('busy');engine.reset();emit({type:'traced',id,result:await engine.trace(data.token)});}
  else if(op==='kernel-check'){await ready;if(busy)throw Error('busy');emit({type:'checked',id,result:await engine.kernel_check()});}
  else if(op==='tokenize-check'){await ready;if(busy)throw Error('busy');emit({type:'checked',id,result:data.texts.map(text=>Array.from(engine.encode(text)))});}
  else if(op==='equivalence-check'){
   await ready;if(busy)throw Error('busy');const ids=Array.from(engine.encode(data.text)).slice(0,data.limit??40);engine.reset();engine.set_reference(true);const expected=[];
   for(const token of ids)expected.push(await engine.logits(token));
   engine.reset();engine.set_reference(false);let max=0,count=0;
   for(let j=0;j<ids.length;j++){const actual=await engine.logits(ids[j]);for(let i=0;i<actual.length;i++){max=Math.max(max,Math.abs(actual[i]-expected[j][i]));if(actual[i]!==expected[j][i])count++;}}
   engine.reset();cached=[];emit({type:'checked',id,result:{tokens:ids.length,logits:ids.length*151669,max_abs_error:max,unequal_logits:count}});
  }
  else if(op==='candidate-check'){
   await ready;if(busy)throw Error('busy');engine.reset();const token=Array.from(engine.encode('Hello'))[0];const logits=await engine.logits(token);
   const expected=Array.from(logits,(score,id)=>[id,score]).sort((a,b)=>b[1]-a[1]||b[0]-a[0]);const checks=[];
   for(const k of [1,20,64]){engine.reset();const actual=await engine.forward(new Uint32Array([token]),k);checks.push({top_k:k,exact:actual.every((p,i)=>p[0]===expected[i][0]&&p[1]===expected[i][1])});}
   engine.reset();cached=[];emit({type:'checked',id,result:checks});
  }
  else if(op==='batch-check'){
   await ready;if(busy)throw Error('busy');const ids=Array.from(engine.encode(data.text)).slice(0,data.limit??40);engine.reset();let single;
   for(const token of ids)single=await engine.logits(token);
   engine.reset();let batched;for(let i=0;i<ids.length;i+=loadInfo.batch)batched=await engine.logits_chunk(new Uint32Array(ids.slice(i,i+loadInfo.batch)));
   let max=0,count=0;for(let i=0;i<single.length;i++){max=Math.max(max,Math.abs(single[i]-batched[i]));if(single[i]!==batched[i])count++;}
   cached=[];engine.reset();emit({type:'checked',id,result:{tokens:ids.length,max_abs_error:max,unequal_logits:count}});
  }
 }catch(e){emit({type:'error',id,message:String(e?.message??e),stack:e?.stack});}
};
