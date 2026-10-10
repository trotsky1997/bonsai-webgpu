import {SvelteMap} from 'svelte/reactivity';
import type {Snapshot,Entry} from './runtime';
class WorkspaceStore {
 snapshots=new SvelteMap<string,Snapshot>();
 open=$state(false);selectedPath=$state('');error=$state('');activeConversation=$state('');
 private channel?:BroadcastChannel;
 private key(conversation:string){return new URL('./',document.baseURI).pathname+':'+conversation;}
 private async request(conversation:string,name:string,args:Record<string,unknown>,id:string,signal?:AbortSignal):Promise<{snapshot?:Snapshot;result:string;isError?:boolean}>{
  if(signal?.aborted)throw new DOMException('Cancelled','AbortError');
  return new Promise((resolve,reject)=>{
   const worker=new Worker(new URL('./worker.ts',import.meta.url),{type:'module'});
   const done=()=>{clearTimeout(timeout);signal?.removeEventListener('abort',abort);worker.terminate();};
   const abort=()=>{done();reject(new DOMException('Cancelled','AbortError'));};
   const timeout=setTimeout(()=>{done();reject(Error('工作区操作超时；已结束 Worker，可重试。'));},12000);
   signal?.addEventListener('abort',abort,{once:true});
   worker.onerror=e=>{done();reject(Error(e.message||'工作区加载失败'));};
   worker.onmessage=({data})=>{if(data.id===id){done();resolve(data);}};
   worker.postMessage({id,key:this.key(conversation),name,args});
  });
 }
 private connect(){
  if(this.channel)return;this.channel=new BroadcastChannel('bonsai-workspace.v1:'+new URL('./',document.baseURI).pathname);
  this.channel.onmessage=({data})=>{if(data.conversation===this.activeConversation)void this.load(data.conversation);};
 }
 async load(conversation:string){
  this.connect();this.activeConversation=conversation;this.error='';
  this.selectedPath=sessionStorage.getItem('bonsai-artifact:'+conversation)??'';
  if(!conversation)return;
  try{const r=await this.request(conversation,'snapshot',{},crypto.randomUUID());if(r.snapshot)this.snapshots.set(conversation,r.snapshot);}catch(e){this.error=String((e as Error)?.message??e);}
 }
 async execute(name:string,args:Record<string,unknown>,conversation:string,id:string,signal?:AbortSignal){
  this.connect();try{
   const before=this.snapshots.get(conversation),r=await this.request(conversation,name,args,id,signal);
   if(r.snapshot){this.snapshots.set(conversation,r.snapshot);this.channel?.postMessage({conversation});
    if(this.activeConversation===conversation){const changed=r.snapshot.entries.filter(e=>e.type==='file'&&e.path.startsWith('/workspace/')&&before?.entries.find(p=>p.path===e.path)?.mtime!==e.mtime);const file=changed.find(e=>/\.html?$/i.test(e.path))??changed.at(-1);if(file)this.select(file.path);}}
   return {content:r.result,isError:!!r.isError};
  }catch(e){if(signal?.aborted)throw e;return {content:'Workspace error: '+String((e as Error)?.message??e),isError:true};}
 }
 async deleteConversation(conversation:string){await this.request(conversation,'delete',{},crypto.randomUUID());this.snapshots.delete(conversation);sessionStorage.removeItem('bonsai-artifact:'+conversation);if(this.activeConversation===conversation){this.open=false;this.selectedPath='';}this.channel?.postMessage({conversation});}
 select(path:string){this.selectedPath=path;this.open=true;sessionStorage.setItem('bonsai-artifact:'+this.activeConversation,path);}
 files(conversation:string):Entry[]{return this.snapshots.get(conversation)?.entries.filter(e=>e.type==='file'&&e.path.startsWith('/workspace/'))??[];}
}
export const workspaceStore=new WorkspaceStore();
