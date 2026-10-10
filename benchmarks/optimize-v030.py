"""Compare v0.3 with v0.2 and native MLX. Open /bench.html and /bench.html?variant=v020."""
import argparse,json,pathlib,subprocess,os,time,urllib.request,hashlib
BASE=pathlib.Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18902');p.add_argument('--model-dir',type=pathlib.Path,default=BASE.parents[1]/'work/model');p.add_argument('--mlx-lib',type=pathlib.Path,default=BASE.parent/'bonsai-mlx-chat/lib');p.add_argument('--rounds',type=int,default=7);args=p.parse_args()
URL=args.url.rstrip('/');RUN=str(time.time_ns());OUT=BASE/'benchmarks/optimization-v030.json'
def fetch(path,body=None):
 req=urllib.request.Request(URL+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=15) as r:return json.load(r)
def browser(variant,op,**params):
 job_id=RUN+'-'+str(time.time_ns());fetch('/benchmark-command',dict(job_id=job_id,variant=variant,op=op,**params));deadline=time.monotonic()+300
 while time.monotonic()<deadline:
  value=fetch('/benchmark-status').get(job_id)
  if value is not None:
   if 'error' in value:raise RuntimeError(value)
   return value
  time.sleep(.1)
 raise TimeoutError(job_id)
def prompt(text):return '<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
env={**os.environ,'BONSAI_BACKEND':'mlx','DYLD_LIBRARY_PATH':str(args.mlx_lib.resolve()),'BONSAI_MLX_METALLIB':str(args.mlx_lib.resolve()/'mlx.metallib'),'BONSAI_CONTEXT':'2048'}
log=(BASE/'benchmarks/optimization-v030-native.log').open('w')
worker=BASE/'benchmarks/bonsai-mlx-benchmark'
native_process=subprocess.Popen([str(worker),'--chat-worker',str(args.model_dir.resolve())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
serial=0
def native(op,**body):
 global serial
 serial+=1;rid=serial;start=time.perf_counter();first=None;text=''
 native_process.stdin.write(json.dumps(dict(id=rid,op=op,**body))+'\n');native_process.stdin.flush()
 while True:
  line=native_process.stdout.readline()
  if not line:raise RuntimeError('Native worker exited')
  e=json.loads(line)
  if e.get('id')!=rid:continue
  if e['type']=='error':raise RuntimeError(e)
  if e['type']=='delta':
   text+=e['text']
   if first is None:first=(time.perf_counter()-start)*1000
  if e['type'] in ['done','reset']:
   return dict(e,content=text,wall_ttft_ms=first,wall_ms=(time.perf_counter()-start)*1000,process_rss_bytes=int(subprocess.check_output(['ps','-o','rss=','-p',str(native_process.pid)],text=True).strip())*1024)
def generate(variant,body):
 if variant=='mlx':native('reset');return native('generate',**body)
 return browser(variant,'generate',body=body)
report={'method':{'baseline':'v0.2.0 d045b275a413f9ea592ba13bbb661f98a05a4cca','candidate':'v0.3.0','hardware':'Apple M5 Pro, 48 GiB, macOS ARM64','browser':'Codex Chromium 155','rounds':args.rounds,'decode_tokens':64,'decode_rate':'63 / decode_seconds','context':2048,'kv':'FP32','browser_kv_capacity_during_performance':512,'browser_prefill_batch':8,'native_prefill_batch':1,'order':'three-way rotation after warmup; prefix reset for each request','load':'fresh Worker, cached WASM/pipelines; cache-cold also forces local HTTP fetch; cache-warm uses CacheStorage'},'performance':[],'greedy':[],'loading':[]}
def save():OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
try:
 report['native_ready']=json.loads(native_process.stdout.readline());assert report['native_ready']['backend']=='mlx-metal',report['native_ready'];report['native_sha256']=hashlib.sha256(worker.read_bytes()).hexdigest()
 warmbody=dict(prompt=prompt(('A browser inference engine keeps compressed weights on the GPU and reuses activation packing across projections. '*12)+'\nSummarize the above in English.'),temperature=0,top_k=1,top_p=1,max_tokens=64,ignore_eos=True,seed=91)
 for v in ['v020','optimized','mlx']:generate(v,warmbody)
 report['browser_info']={v:browser(v,'info') for v in ['v020','optimized']};assert all(report['browser_info'][v]['memory']['kv_capacity_tokens']==512 for v in ['v020','optimized']);save()
 for name,question,sampling in [('short_greedy','Explain in English why the sky is blue, in a short paragraph.',False),('long_greedy',('A browser inference engine keeps compressed weights on the GPU and reuses activation packing across projections. '*12)+'\nSummarize the above in English.',False),('short_sampling','Explain in English why the sky is blue, in a short paragraph.',True)]:
  body=dict(prompt=prompt(question),temperature=.5 if sampling else 0,top_k=20 if sampling else 1,top_p=.85 if sampling else 1,max_tokens=64,ignore_eos=True,seed=91)
  variants=['v020','optimized','mlx']
  for v in variants:generate(v,body)
  for r in range(args.rounds):
   order=variants[r%3:]+variants[:r%3];row=dict(workload=name,round=r,order=order)
   for v in order:row[v]=generate(v,body)
   report['performance'].append(row);save();print(name,r,{v:round(63000/row[v]['decode_ms'],2) for v in variants},flush=True)
 for q in ['你好，请用一句话介绍自己。','用两句话解释为什么天空是蓝色的。','Write a short Python function that returns the square of an integer.']:
  body=dict(prompt=prompt(q),temperature=0,top_k=1,top_p=1,max_tokens=64)
  row=dict(question=q,**{v:generate(v,body) for v in ['v020','optimized','mlx']});row['browser_exact']=row['v020']['tokens']==row['optimized']['tokens'];row['mlx_exact']=row['mlx']['tokens']==row['optimized']['tokens'];report['greedy'].append(row);save()
 # Alternate variants to avoid measuring all warm loads in only one order.
 for cold in [False,True]:
  rounds=3 if cold else 7
  for r in range(rounds):
   for v in (['v020','optimized'] if r%2==0 else ['optimized','v020']):
    value=browser(v,'load-check',rounds=1,cold=cold)['result'][0];report['loading'].append(dict(cold=cold,variant=v,round=r,**value));save();print('load',cold,v,r,round(value['load_ms'],2),flush=True)
 report['final_info']={v:browser(v,'info') for v in ['v020','optimized']};save();print('COMPLETE',flush=True)
finally:
 native_process.terminate();native_process.wait(timeout=20);log.close()
