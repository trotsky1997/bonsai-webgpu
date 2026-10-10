import json,subprocess,time,os,pathlib,urllib.request,statistics,hashlib,argparse
BASE=pathlib.Path(__file__).resolve().parent.parent
ROOT=BASE.parents[1]
parser=argparse.ArgumentParser(description='Paired native MLX / browser WebGPU comparison; open /bench.html before running.')
parser.add_argument('--url',default='http://127.0.0.1:18902')
parser.add_argument('--native-worker',type=pathlib.Path,default=BASE/'benchmarks/bonsai-mlx-benchmark')
parser.add_argument('--mlx-lib',type=pathlib.Path,default=BASE.parent/'bonsai-mlx-chat/lib')
parser.add_argument('--model-dir',type=pathlib.Path,default=ROOT/'work/model')
parser.add_argument('--output-dir',type=pathlib.Path,default=BASE/'benchmarks')
parser.add_argument('--rounds',type=int,default=7)
args=parser.parse_args()
OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
URL=args.url.rstrip('/')
RUN=str(time.time_ns())
env={**os.environ,'BONSAI_BACKEND':'mlx','BONSAI_MLX_METALLIB':str(args.mlx_lib.resolve()/'mlx.metallib'),'DYLD_LIBRARY_PATH':str(args.mlx_lib.resolve()),'BONSAI_CONTEXT':'2048'}
log=(OUT/'native-worker.log').open('w')
p=subprocess.Popen([str(args.native_worker.resolve()),'--chat-worker',str(args.model_dir.resolve())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,env=env)
serial=0
report={'method':{'hardware':'Apple M5 Pro, macOS ARM64','context_limit':2048,'kv_precision':'FP32','activation':'W1A4 g32 sign magnitude [-7,7]','weight_group':128,'decode_tokens':64,'rounds':args.rounds,'browser_prefill_batch':8,'mlx_prefill_batch':1,'timing':'TTFT is wall time to first text delta; decode rate = (output tokens - 1) / decode phase seconds; prefix cache reset for every request.'},'quality':[],'greedy':[],'performance':[]}
def save(): (OUT/'comparison-raw.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
def native(op,**args):
 global serial
 serial+=1; rid=serial;start=time.perf_counter();first=None;text=''
 p.stdin.write(json.dumps({'id':rid,'op':op,**args},ensure_ascii=False)+'\n');p.stdin.flush()
 while True:
  line=p.stdout.readline()
  if not line:raise RuntimeError('native exited: '+str(p.poll()))
  e=json.loads(line)
  if e.get('type')=='ready':continue
  if e.get('id')!=rid:continue
  if e['type']=='error':raise RuntimeError(e)
  if e['type']=='delta':
   text+=e['text']
   if first is None:first=(time.perf_counter()-start)*1000
  if e['type'] in ('done','reset','diagnosed','info','tokens'):
   e.update(content=text,wall_ttft_ms=first,wall_ms=(time.perf_counter()-start)*1000)
   e['process_rss_bytes']=int(subprocess.check_output(['ps','-o','rss=','-p',str(p.pid)],text=True).strip())*1024
   return e

def fetch(path,body=None):
 req=urllib.request.Request(URL+path,data=json.dumps(body).encode() if body is not None else None,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=15) as r:return json.load(r)
def browser(op,job_id,**args):
 job_id=RUN+'-'+job_id
 fetch('/benchmark-command',{'op':op,'job_id':job_id,**args})
 deadline=time.monotonic()+240
 while time.monotonic()<deadline:
  values=fetch('/benchmark-status')
  if job_id in values:
   if 'error' in values[job_id]:raise RuntimeError(values[job_id])
   return values[job_id]
  time.sleep(.2)
 raise TimeoutError(job_id)
def prompt(q):return '<|im_start|>user\n'+q+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
def ng(body):native('reset');return native('generate',**body)
def bg(body,name):return browser('generate',name,body=body)
try:
 ready=json.loads(p.stdout.readline());assert ready['backend']=='mlx-metal',ready;report['native_ready']=ready;report['native_worker_sha256']=hashlib.sha256(args.native_worker.read_bytes()).hexdigest();print('native ready',ready,flush=True)
 report['browser_info']=browser('info','initial-info');save()
 texts=[
 ('diagnostic','english','A language model predicts the next token from the preceding context. Quantizing weights reduces memory traffic, while activation quantization changes the arithmetic used by each projection. Accurate evaluation compares the same token sequence under both implementations.'),
 ('diagnostic','chinese','低比特推理需要同时考虑权重带宽、激活打包和计算调度。模型的权重保持一位，激活按三十二个元素分组量化为四位整数。注意力和归一化继续使用浮点计算。'),
 ('diagnostic','code','fn sum(values: &[i32]) -> i32 {\n    values.iter().copied().sum()\n}\n\nassert_eq!(sum(&[1, 2, 3]), 6);'),
 ('holdout','science','Water evaporates from the ocean and condenses into clouds. When droplets become large enough, they fall as rain. This cycle transports heat and supplies fresh water to ecosystems on land.'),
 ('holdout','reasoning','There are twelve apples in a basket. Alice takes three apples and Bob adds five. The basket now contains fourteen apples. If those apples are divided equally between two children, each child receives seven.'),
 ('holdout','chinese','春天的清晨，河边的小路上有几个人正在散步。阳光透过树叶照在水面上，远处传来鸟叫声。一个孩子停下来观察石头旁边的蚂蚁，然后继续向前走。')]
 for group,filename in [('diagnostic','a2_diagnostic_texts.jsonl'),('holdout','sidecar_calibration_holdout.jsonl')]:
  for line in (BASE/'benchmarks/fixtures'/filename).read_text().splitlines():
   item=json.loads(line);texts.append((group,item['name'],item['text']))
 for group,name,text in texts:
  n=native('diagnose',text=text);b=browser('diagnose','quality-'+group+'-'+name,job={'tokens':n['result']['tokens']})
  nr=n['result'];br=b['result'];row={'group':group,'name':name,'text':text,'native':nr,'browser':br,'mean_nll_delta':br['mean_nll']-nr['mean_nll'],'top1_agreement':sum(a==c for a,c in zip(nr['predictions'],br['predictions']))/len(nr['predictions']),'max_target_logit_error':max(abs(a-c) for a,c in zip(nr['target_logits'],br['target_logits']))};report['quality'].append(row);save();print('quality',name,row['mean_nll_delta'],row['top1_agreement'],flush=True)
 for i,q in enumerate(['你好，请用一句话介绍自己。','用两句话解释为什么天空是蓝色的。','Write a short Python function that returns the square of an integer.']):
  body={'prompt':prompt(q),'temperature':0,'top_k':1,'top_p':1,'max_tokens':64}
  n=ng(body);b=bg(body,'greedy-'+str(i));a=n['tokens'];c=b['tokens'];prefix=0
  for x,y in zip(a,c):
   if x!=y:break
   prefix+=1
  report['greedy'].append({'question':q,'native':n,'browser':b,'exact':a==c,'common_prefix':prefix});save();print('greedy',i,a==c,prefix,len(a),len(c),flush=True)
 prompts=[('short',prompt('Explain in English why the sky is blue, in a short paragraph.')),('long',prompt(('A browser inference engine keeps compressed weights on the GPU and reuses activation packing across projections. '*12)+'\nSummarize the above in English.'))]
 for name,pr in prompts:
  body={'prompt':pr,'temperature':0,'top_k':1,'top_p':1,'max_tokens':64,'ignore_eos':True}
  ng(body);bg(body,'warm-'+name)
  for r in range(args.rounds):
   if r%2==0:n=ng(body);b=bg(body,f'perf-{name}-{r}')
   else:b=bg(body,f'perf-{name}-{r}');n=ng(body)
   report['performance'].append({'prompt_name':name,'round':r,'order':['mlx','browser'] if r%2==0 else ['browser','mlx'],'native':n,'browser':b});save();print('perf',name,r,round(63e3/n['decode_ms'],2),round(63e3/b['decode_ms'],2),flush=True)
 report['browser_final']=browser('info','final-info');save()
 print('COMPLETE',flush=True)
finally:
 p.terminate();p.wait(timeout=20);log.close()
