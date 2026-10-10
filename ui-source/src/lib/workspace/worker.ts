/// <reference lib="webworker" />
import './globals.browser';
import {createWorkspace,type Snapshot} from './runtime';
const dbPromise=new Promise<IDBDatabase>((resolve,reject)=>{const r=indexedDB.open('bonsai-workspace.v1:'+new URL('../',self.location.href).origin,1);r.onupgradeneeded=()=>r.result.createObjectStore('workspaces');r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});
async function load(key:string){const db=await dbPromise;return new Promise<Snapshot|undefined>((resolve,reject)=>{const r=db.transaction('workspaces').objectStore('workspaces').get(key);r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);});}
async function save(key:string,snapshot:Snapshot,base:number){const db=await dbPromise;await new Promise<void>((resolve,reject)=>{const tx=db.transaction('workspaces','readwrite'),s=tx.objectStore('workspaces'),r=s.get(key);let conflict=false;r.onsuccess=()=>{if((r.result?.version??0)!==base){conflict=true;tx.abort();}else s.put(snapshot,key);};tx.oncomplete=()=>resolve();tx.onabort=tx.onerror=()=>reject(Error(conflict?'Workspace changed in another tab. Retry.':String(tx.error)));});}
self.onmessage=async({data})=>{
 const run=async()=>{
  if(data.name==='delete'){const db=await dbPromise;await new Promise<void>((resolve,reject)=>{const tx=db.transaction('workspaces','readwrite');tx.objectStore('workspaces').delete(data.key);tx.oncomplete=()=>resolve();tx.onerror=()=>reject(tx.error);});return {result:''};}
  const old=await load(data.key),workspace=await createWorkspace(old);
  if(data.name==='import'&&old?.imports?.includes(data.id))return {snapshot:old,result:''};
  const prior=old?.operations.find(o=>o.id===data.id);
  if(data.name==='snapshot')return {snapshot:old??{version:0,entries:await workspace.serialize(),imports:[],operations:[]},result:''};
  if(prior)return {snapshot:old,result:prior.result,isError:prior.isError};
  let result:string,isError=false;
  try{result=await workspace.execute(data.name==='import'?'write':data.name,data.args??{});}catch(e){result='Workspace error: '+String((e as Error)?.message??e);isError=true;}
  const snapshot:Snapshot={version:(old?.version??0)+1,entries:await workspace.serialize(),imports:[...(old?.imports??[]),...(data.name==='import'&&!isError?[data.id]:[])].slice(-4096),operations:[...(old?.operations??[]),{id:data.id,result,isError}].slice(-128)};
  await save(data.key,snapshot,old?.version??0);return {snapshot,result,isError};
 };
 try{const reply=navigator.locks?await navigator.locks.request('bonsai-workspace:'+data.key,run):await run();postMessage({id:data.id,...reply});}catch(e){postMessage({id:data.id,result:'Workspace error: '+String((e as Error)?.message??e),isError:true});}
};
