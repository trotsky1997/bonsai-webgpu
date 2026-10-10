"""Validate the actual browser kernels; open /bench.html first."""
import argparse,json,pathlib,time,urllib.request,random,struct
P=argparse.ArgumentParser();P.add_argument('--url',default='http://127.0.0.1:18902');args=P.parse_args()
root=pathlib.Path(__file__).resolve().parent
def rpc(op,**params):
    job={'job_id':str(time.time_ns()),'op':op,**params}
    req=urllib.request.Request(args.url+'/benchmark-command',data=json.dumps(job).encode(),headers={'Content-Type':'application/json'})
    urllib.request.urlopen(req).close()
    for _ in range(1200):
        with urllib.request.urlopen(args.url+'/benchmark-status') as res:values=json.load(res)
        if job['job_id'] in values:
            result=values[job['job_id']]
            if 'error' in result:raise RuntimeError(result)
            return result['result']
        time.sleep(.1)
    raise TimeoutError(op)
result={'kernel':rpc('kernel-check'),'candidates':rpc('candidate-check')}
text='A browser inference engine keeps compressed weights on the GPU and reuses activation packing across projections. '*30
result['fused_equivalence']=rpc('equivalence-check',text=text,limit=40)
result['batch40']=rpc('batch-check',text=text,limit=40)
result['batch320']=rpc('batch-check',text=text,limit=320)
cases=json.loads((root/'comparison-raw.json').read_text())['quality']
tokens=rpc('tokenize-check',texts=[x['text'] for x in cases])
result['tokenizer']={'cases':len(cases),'exact':tokens==[x['native']['tokens'] for x in cases]}
tokenizer=json.loads((root.parent/'model/tokenizer.json').read_bytes())
edges=['Cafe\u0301 Café','👨‍👩‍👧‍👦🙂🚀','中文\n\t  空格','a\r\nb\n\n c','1234567890 -0.5 1e-9',"I'm you're we've DON'T",'αβγ éß русский العربية','零宽\u200b测试']+[t['content'] for t in tokenizer['added_tokens']]
a=rpc('tokenize-check',texts=edges)
b=rpc('tokenize-check',variant='v020',texts=edges)
result['tokenizer_edges']={'cases':len(edges),'exact':a==b}

generator=random.Random(91);failures=0
for _ in range(1000000):
    bits=generator.randrange(0x7f800000);exponent=bits>>23;mantissa=(bits&0x7fffff)|0x800000
    if exponent<=3:
        numerator=((bits&0x7fffff) if exponent==0 else mantissa)<<max(exponent-1,0)
        value=numerator//7+(numerator%7>=4)
    else:
        shift=2 if mantissa>=14680064 else 3;numerator=mantissa<<shift
        value=((exponent-shift)<<23)|((numerator//7+(numerator%7>=4))-0x800000)
    x=struct.unpack('<f',struct.pack('<I',bits))[0]
    expected=struct.unpack('<I',struct.pack('<f',x/7))[0]
    failures+=value!=expected
result['scale_integer_oracle']={'cases':1000000,'bit_errors':failures}
(root/'checks-v030.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert result['kernel']['bit_errors']==0 and result['kernel']['scale_errors']==0 and result['kernel']['linear_errors']==0
assert all(x['exact'] for x in result['candidates']) and result['tokenizer']['exact']
assert result['fused_equivalence']['max_abs_error']==0 and result['tokenizer_edges']['exact']
assert result['batch40']['max_abs_error']==0 and result['batch320']['max_abs_error']==0 and failures==0
