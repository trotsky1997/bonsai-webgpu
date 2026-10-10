// OpenAI-shaped streaming transport, entirely within this browser tab.
const nativeFetch=window.fetch.bind(window);
const appBase=new URL('../',import.meta.url);
const worker=new Worker(new URL('web/worker.mjs?v=0.4.1.2',appBase),{type:'module'});
let resolveReady,rejectReady,info,serial=0;
const ready=new Promise((resolve,reject)=>{resolveReady=resolve;rejectReady=reject;});
const listeners=new Map();
const overlay=document.createElement('div');overlay.id='bonsai-loading';
overlay.style.cssText='position:fixed;inset:0;background:rgba(20,23,29,.97);color:#eee;z-index:99999;display:flex;align-items:center;justify-content:center;font:16px system-ui;text-align:center';
overlay.innerHTML='<div><h2>Bonsai · 浏览器本地聊天</h2><p id="bonsai-status">正在准备模型…</p><p style="font-size:13px;color:#aaa">首次加载约 242 MB；权重会缓存在此浏览器。</p><progress max="100" value="0" style="width:280px"></progress></div>';
document.documentElement.append(overlay);
function loadFailed(message){
 rejectReady(Error(message));overlay.querySelector('#bonsai-status').textContent=message;
 if(overlay.querySelector('[data-recovery]'))return;
 const recovery=document.createElement('div');recovery.dataset.recovery='';
 const hint=document.createElement('p');hint.textContent='GPU 缓冲区不足时，可缩短上下文后重试。';recovery.append(hint);
 for(const context of [32768,4096]){
  const button=document.createElement('button');button.textContent=`使用 ${context===32768?'32K':'4K'} · INT4 重试`;
  button.style.cssText='min-height:44px;margin:6px;padding:10px 16px;cursor:pointer';
  button.onclick=()=>{localStorage.setItem('bonsai-runtime',JSON.stringify({...runtime,context,kv_bits:4}));location.reload();};recovery.append(button);
 }
 overlay.firstElementChild.append(recovery);
}
worker.onmessage=({data})=>{
 if(data.type==='loading'){overlay.querySelector('#bonsai-status').textContent=data.message;overlay.querySelector('progress').value=data.percent;}
 else if(data.type==='ready'){info=data;window.bonsaiInfo=data;resolveReady(data);overlay.remove();}
 else if(data.type==='prefix-ready'||data.type==='prefix-progress'){window.bonsaiPrefixInfo=data;window.dispatchEvent(new CustomEvent('bonsai-prefix-status',{detail:data}));}
 else if(data.type==='error'&&!data.id){loadFailed(data.message);console.error(data);}
 else listeners.get(data.id)?.(data);
};
worker.onerror=e=>loadFailed(e.message);
let runtime={};try{runtime=JSON.parse(localStorage.getItem('bonsai-runtime')??'{}')??{};}catch{}
worker.postMessage({op:'init',context:runtime.context??131072,kv_bits:runtime.kv_bits??4});
let warmTimer,warmConfiguration='';
window.addEventListener('bonsai-prefix-config',event=>{
 if(runtime.prefix_cache===false)return;
 const body={...event.detail,prefill_attention:runtime.prefill_attention??'flashprefill'};
 const configuration=JSON.stringify(body);if(configuration===warmConfiguration)return;
 warmConfiguration=configuration;clearTimeout(warmTimer);
 warmTimer=setTimeout(async()=>{try{await ready;if(configuration===warmConfiguration&&!listeners.size)worker.postMessage({op:'warm-prefix',id:++serial,body});}catch{}},250);
});
const json=(body,status=200)=>new Response(JSON.stringify(body),{status,headers:{'Content-Type':'application/json'}});
function props(){return {role:'model',model_path:'Bonsai-1.7B-W1A4-WebGPU',total_slots:1,default_generation_settings:{id:0,n_ctx:info?.context??131072,is_processing:false,prompt:'',params:{temperature:0.5,top_k:20,top_p:0.85,min_p:0,max_tokens:1024,n_predict:1024,repeat_penalty:1,repeat_last_n:64,presence_penalty:0,frequency_penalty:0,seed:-1,stop:[],samplers:['top_k','top_p','temperature'],reasoning_format:'none'}},modalities:{vision:false,audio:false,video:false},chat_template:'',bos_token:'',eos_token:'<|im_end|>',build_info:'Bonsai Rust/WASM 0.4 · 浏览器 WebGPU · W1A4',ui_settings:{titleGenerationUseLLM:false,preEncodeConversation:false,showMessageStats:true,systemMessage:'',symbolicMathEnabled:false,jsSandboxEnabled:false},cors_proxy_enabled:false};}
window.fetch=async(input,options={})=>{
 const req=new Request(input,options);const url=new URL(req.url);if(url.origin!==location.origin)return nativeFetch(req);
 const path=url.pathname.startsWith(appBase.pathname)?'/'+url.pathname.slice(appBase.pathname.length):url.pathname;
 if(path==='/props')return json(props());
 if(path==='/v1/models')return json({object:'list',data:[{id:'Bonsai-1.7B-W1A4-WebGPU',object:'model',created:0,owned_by:'prism-ml',meta:{n_ctx_train:32768,n_vocab:151669}}]});
 if(path==='/slots')return json([{id:0,n_ctx:info?.context??131072,is_processing:false}]);
 if(path==='/tools'||path==='/v1/streams/lookup')return json([]);
 if(path==='/health'){await ready;return json({...info,status:'ok'});}
 if(path==='/v1/stream'&&req.method==='DELETE'){worker.postMessage({op:'cancel'});return json({success:true});}
 if(path!=='/v1/chat/completions')return nativeFetch(req);
 await ready;const body=await req.json();body.prefill_attention??=runtime.prefill_attention??'flashprefill';body.prefix_cache??=runtime.prefix_cache??true;if(body.n_predict===0)return json({choices:[]});
 const id=++serial,completionId='chatcmpl-'+crypto.randomUUID(),created=Math.floor(Date.now()/1000),encoder=new TextEncoder();
 const envelope=payload=>({id:completionId,object:'chat.completion.chunk',created,model:'Bonsai-1.7B-W1A4-WebGPU',...payload});
 let whole='',toolCalls=[];
 const stream=new ReadableStream({start(controller){
  const send=payload=>controller.enqueue(encoder.encode('data: '+JSON.stringify(envelope(payload))+'\n\n'));
  send({choices:[{index:0,delta:{role:'assistant',content:''},finish_reason:null}]});
  listeners.set(id,event=>{
   if(event.type==='delta'){whole+=event.text;send({choices:[{index:0,delta:{content:event.text},finish_reason:null}]});}
   else if(event.type==='tool_calls'){const calls=event.tool_calls.map(call=>{if(body.client_capabilities?.includes('interactive-ui.v1'))return call;const {ui_offset,...plain}=call;return plain;});for(const call of calls)toolCalls[call.index]=call;send({choices:[{index:0,delta:{tool_calls:calls},finish_reason:null}]});}
   else if(event.type==='progress')send({choices:[],prompt_progress:{total:event.prompt_tokens,cache:event.cached_tokens,processed:event.processed_tokens,time_ms:0}});
   else if(event.type==='done'){
    const usage={prompt_tokens:event.prompt_tokens,completion_tokens:event.completion_tokens,total_tokens:event.prompt_tokens+event.completion_tokens};
    const timings={prompt_n:event.prompt_tokens-event.cached_tokens,prompt_total_n:event.prompt_tokens,prompt_ms:event.prefill_ms,prompt_per_second:(event.prompt_tokens-event.cached_tokens)*1000/Math.max(event.prefill_ms,1),cache_n:event.cached_tokens,predicted_n:event.completion_tokens,predicted_ms:event.decode_ms,predicted_per_second:Math.max(0,event.completion_tokens-1)*1000/Math.max(event.decode_ms,1),ttft_ms:event.ttft_ms,ngram:event.ngram,prefix_cache:event.prefix_cache,tool_json_repairs:event.tool_json_repairs};
    send({choices:[{index:0,delta:{},finish_reason:event.finish_reason}],usage,timings});controller.enqueue(encoder.encode('data: [DONE]\n\n'));controller.close();listeners.delete(id);
   }else if(event.type==='error'){send({error:{message:event.message}});controller.enqueue(encoder.encode('data: [DONE]\n\n'));controller.close();listeners.delete(id);}
  });
  req.signal.addEventListener('abort',()=>worker.postMessage({op:'cancel',id}),{once:true});
  worker.postMessage({op:'generate',id,body});
 },cancel(){listeners.delete(id);worker.postMessage({op:'cancel',id});}});
 if(body.stream)return new Response(stream,{headers:{'Content-Type':'text/event-stream','Cache-Control':'no-cache'}});
 const reader=stream.getReader();let last;while(true){const {value,done}=await reader.read();if(done)break;for(const line of new TextDecoder().decode(value).split('\n'))if(line.startsWith('data: {')){const e=JSON.parse(line.slice(6));if(e.error)return json(e,400);if(e.usage)last=e;}}
 return json({id:completionId,object:'chat.completion',created,model:'Bonsai-1.7B-W1A4-WebGPU',choices:[{index:0,message:{role:'assistant',content:whole||null,...(toolCalls.length?{tool_calls:toolCalls.map(({index,...call})=>call)}:{})},finish_reason:last?.choices[0]?.finish_reason}],usage:last?.usage,timings:last?.timings});
};

if('serviceWorker' in navigator)navigator.serviceWorker.register(new URL('sw.js',appBase),{scope:appBase.pathname}).catch(console.error);
