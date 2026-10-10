import {renderSystemPrefix} from './chat-template.mjs?v=0.4.1.2';
const MiB=1024*1024, MAX_ENTRY=64*MiB;
const hash=async bytes=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),n=>n.toString(16).padStart(2,'0')).join('');
const equal=(a,b)=>a.length===b.length&&a.every((n,i)=>n===b[i]);
export function commonPrefix(a,b,limit=b.length){let n=0;while(n<Math.min(a.length,b.length,limit)&&a[n]===b[n])n++;return n;}

// Store only the initial system block. Conversation messages and tool results never
// enter a snapshot. Failure/quota denial simply falls back to normal prefill.
export class PrefixStore {
 constructor(){this.disabled=typeof indexedDB==='undefined';}
 async db(){
  if(this.disabled)return null;
  try{return await(this.promise??=new Promise((resolve,reject)=>{
   const r=indexedDB.open('bonsai-system-kv.v1',1);
   r.onupgradeneeded=()=>{r.result.createObjectStore('payloads');r.result.createObjectStore('metadata');};
   r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);r.onblocked=()=>reject(Error('KV cache database blocked'));
  }));}catch{this.disabled=true;return null;}
 }
 async get(key){const db=await this.db();if(!db)return null;return new Promise((resolve,reject)=>{const r=db.transaction('payloads').objectStore('payloads').get(key);r.onsuccess=()=>resolve(r.result??null);r.onerror=()=>reject(r.error);});}
 async put(key,value){
  const db=await this.db();if(!db)return false;
  return new Promise((resolve,reject)=>{
   const tx=db.transaction(['payloads','metadata'],'readwrite'),meta=tx.objectStore('metadata'),payload=tx.objectStore('payloads');
   const r=meta.getAll();r.onsuccess=()=>{
    const entries=r.result.filter(e=>e.key!==key).sort((a,b)=>b.time-a.time);let bytes=value.bytes.byteLength,count=1;
    for(const e of entries){if(count>=3||bytes+e.size>128*MiB){meta.delete(e.key);payload.delete(e.key);}else{bytes+=e.size;count++;}}
    meta.put({key,size:value.bytes.byteLength,time:Date.now()},key);payload.put(value,key);
   };
   tx.oncomplete=()=>resolve(true);tx.onerror=()=>reject(tx.error);tx.onabort=()=>reject(tx.error);
  });
 }
 async delete(key){const db=await this.db();if(!db)return;await new Promise((resolve,reject)=>{const tx=db.transaction(['payloads','metadata'],'readwrite');for(const name of ['payloads','metadata'])tx.objectStore(name).delete(key);tx.oncomplete=resolve;tx.onerror=()=>reject(tx.error);});}
}

export class PrefixCache {
 constructor(engine,configuration,store=new PrefixStore()){
  this.engine=engine;this.config=configuration;this.store=store;this.entries=new Map();this.plans=new Map();this.clock=0;
  this.bytesPerToken=28*(configuration.kv_bits===4?1280:8192);
 }
 async plan(body,ids,attention){
  if(body.prompt!==undefined||body.prefix_cache===false)return null;
  const text=renderSystemPrefix(body),memoKey=attention+'\0'+text,memo=this.plans.get(memoKey);
  if(memo&&memo.count<ids.length&&commonPrefix(memo.tokens,ids,memo.count)===memo.count)return memo;
  const system=Array.from(this.engine.encode(text));
  // Leave a boundary token to ordinary prefill: BPE at the following message may
  // differ. Compare every selected token against the complete rendered prompt.
  const limit=Math.min(2048,this.config.context-1,Math.floor(MAX_ENTRY/this.bytesPerToken),system.length-1,ids.length-1);
  const count=Math.floor(commonPrefix(system,ids,limit)/8)*8;
  if(count<8)return null;
  const tokens=ids.slice(0,count),configuration={abi:'bonsai-system-kv-1',...this.config,attention};
  const key=await hash(new TextEncoder().encode(JSON.stringify({configuration,tokens})));
  const plan={key,tokens,count};if(this.plans.size>=8)this.plans.delete(this.plans.keys().next().value);this.plans.set(memoKey,plan);return plan;
 }
 allocate(plan){
  let slot=[0,1].find(n=>![...this.entries.values()].some(e=>e.slot===n));
  if(slot===undefined){const oldest=[...this.entries.values()].sort((a,b)=>a.used-b.used)[0];slot=oldest.slot;this.entries.delete(oldest.key);this.engine.drop_prefix(slot);}
  const entry={...plan,slot,used:++this.clock,persisted:false,origin:'gpu-prefix'};this.entries.set(plan.key,entry);return entry;
 }
 async find(plan){
  if(!plan)return null;
  let entry=this.entries.get(plan.key);if(entry){entry.used=++this.clock;return entry;}
  try{
   const saved=await this.store.get(plan.key);if(!saved)return null;
   if(!equal(saved.tokens??[],plan.tokens)||!(saved.bytes instanceof Uint8Array)||saved.bytes.byteLength!==plan.count*this.bytesPerToken||saved.sha256!==await hash(saved.bytes))throw Error('invalid system KV cache');
   entry=this.allocate(plan);this.engine.import_prefix(entry.slot,plan.count,this.config.kv_bits,saved.bytes);entry.persisted=true;entry.origin='disk-prefix';return entry;
  }catch{if(entry){this.entries.delete(entry.key);this.engine.drop_prefix(entry.slot);}await this.store.delete(plan.key).catch(()=>{});return null;}
 }
 capture(plan){
  if(!plan)return null;
  let entry=this.entries.get(plan.key);if(entry)return entry;
  entry=this.allocate(plan);
  try{this.engine.save_prefix(entry.slot,plan.count);return entry;}catch(e){this.entries.delete(entry.key);this.engine.drop_prefix(entry.slot);throw e;}
 }
 restore(entry){this.engine.restore_prefix(entry.slot);entry.used=++this.clock;const source=entry.origin;entry.origin='gpu-prefix';return source;}
 async persist(entry){
  if(!entry)return false;if(entry.persisted)return true;if(this.store.disabled)return false;
  try{const bytes=this.engine.export_prefix?await this.engine.export_prefix(entry.slot):null;if(!bytes)return false;
   entry.persisted=await this.store.put(entry.key,{tokens:entry.tokens,bytes,sha256:await hash(bytes)});return entry.persisted;
  }catch{this.store.disabled=true;return false;}
 }
 clear(){for(const entry of this.entries.values())this.engine.drop_prefix(entry.slot);this.entries.clear();}
}
