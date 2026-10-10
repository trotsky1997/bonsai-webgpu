#!/usr/bin/env python3
"""Offline layout conversion only; browser inference has no Python dependency."""
import argparse, json, pathlib, struct, hashlib
import numpy as np

def pack_tokenizer(source,dest):
    tokenizer=json.loads((source/'tokenizer.json').read_bytes())
    vocab=tokenizer['model']['vocab'];tokens=[None]*len(vocab)
    for token,index in vocab.items():tokens[index]=token.encode()
    merges=tokenizer['model']['merges'];tokenizer['model']['vocab']={};tokenizer['model']['merges']=[]
    header=json.dumps(tokenizer,ensure_ascii=False,separators=(',',':')).encode()
    data=bytearray(b'BNSITOK1'+struct.pack('<I',len(header))+header+struct.pack('<I',len(tokens)))
    for token in tokens:data+=struct.pack('<I',len(token))+token
    data+=struct.pack('<I',len(merges))
    for a,b in merges:data+=struct.pack('<II',vocab[a],vocab[b])
    (dest/'tokenizer.bin').write_bytes(data)
    return {'name':'tokenizer.bin','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def main():
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args()
    source=pathlib.Path(a.source);dest=pathlib.Path(a.destination);dest.mkdir(parents=True,exist_ok=True)
    model=source/'model.safetensors'
    with model.open('rb') as f:
        hlen=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(hlen))
    raw=np.memmap(model,mode='r',dtype=np.uint8,offset=8+hlen)
    def tensor(name):
        t=header[name];start,end=t['data_offsets'];dt={'U32':'<u4','F16':'<f2'}[t['dtype']]
        return np.frombuffer(raw[start:end],dtype=dt).reshape(t['shape'])
    manifest={'format':'bonsai-w1a4-webgpu-v1','weight_bits':1,'activation_bits':4,'activation_group':32,'weight_group':128,'files':[]}
    def blob(filename,matrices,vectors):
        meta={'matrices':[],'vectors':[]};payload=bytearray()
        def put(array):
            b=array.tobytes();start=len(payload);payload.extend(b);return [start,len(payload)]
        for name,path in matrices:
            w=tensor(path+'.weight');s=tensor(path+'.scales');bias=tensor(path+'.biases')
            if not np.array_equal(s.astype(np.float32)*np.float32(-0.5),bias.astype(np.float32)):
                raise ValueError(path+' is not exactly symmetric')
            n,words=w.shape
            meta['matrices'].append({'name':name,'n':n,'k':words*32,'w':put(np.ascontiguousarray(w.T)),'s':put(np.ascontiguousarray(s.T))})
        for name,value in vectors:
            if isinstance(value,str):value=tensor(value).astype('<f4')
            meta['vectors'].append({'name':name,'data':put(np.asarray(value,dtype='<f4'))})
        h=json.dumps(meta,separators=(',',':')).encode();h+=b' '*((-len(h))%8)
        data=struct.pack('<Q',len(h))+h+payload;(dest/filename).write_bytes(data)
        manifest['files'].append({'name':filename,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        print(filename,len(data),flush=True)
    c=json.loads((source/'config.json').read_text());half=c['head_dim']//2
    corr=lambda rotations:c['head_dim']*np.log(c['rope_scaling']['original_max_position_embeddings']/(rotations*2*np.pi))/(2*np.log(c['rope_theta']))
    lo=max(0,min(half-1,np.floor(corr(32))));hi=max(0,min(half-1,np.ceil(corr(1))))
    i=np.arange(half,dtype=np.float32);inv=np.power(np.float32(c['rope_theta']),-(2*i)/np.float32(c['head_dim']))
    ramp=np.clip((i-np.float32(lo))/np.float32(hi-lo if hi!=lo else 0.001),0,1)
    freq=inv/np.float32(c['rope_scaling']['factor'])*ramp+inv*(1-ramp)
    blob('base.bin',[('embed','model.embed_tokens')],[('norm','model.norm.weight'),('freq',freq)])
    for i in range(c['num_hidden_layers']):
        prefix=f'model.layers.{i}'
        matrices=[(n,f'{prefix}.self_attn.{n}_proj') for n in ['q','k','v','o']]+[(n,f'{prefix}.mlp.{n}_proj') for n in ['gate','up','down']]
        vectors=[('input_norm',f'{prefix}.input_layernorm.weight'),('post_norm',f'{prefix}.post_attention_layernorm.weight'),('q_norm',f'{prefix}.self_attn.q_norm.weight'),('k_norm',f'{prefix}.self_attn.k_norm.weight')]
        blob(f'layer-{i:02}.bin',matrices,vectors)
    for name in ['tokenizer.json','config.json']:(dest/name).write_bytes((source/name).read_bytes())
    manifest['tokenizer']=pack_tokenizer(source,dest)
    manifest['checkpoint_sha256']=hashlib.file_digest(model.open('rb'),'sha256').hexdigest()
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
