/** Serialized, cancellable local image processing. Image bytes stay in this browser. */
let worker: Worker | undefined;
let queue: Promise<unknown> = Promise.resolve();
let nextId = 0;
export interface ImageText { text: string; lines: number; elapsedMs: number; version: '0.11.4'; cached?: boolean; imageSha256?: string }
export function recognizeImage(preview: string, signal: AbortSignal): Promise<ImageText> {
 const task = queue.catch(() => {}).then(async () => {
  if(signal.aborted) throw new DOMException('Cancelled','AbortError');
  const bytes=await(await fetch(preview)).arrayBuffer();
  if(signal.aborted) throw new DOMException('Cancelled','AbortError');
  return new Promise<ImageText>((resolve,reject)=>{
   const id=++nextId;
   worker ??= new Worker(new URL('./web/ocr/worker.mjs?v=0.4.1',document.baseURI),{type:'module'});
   const cleanup=()=>signal.removeEventListener('abort',abort);
   const fail=(error:unknown)=>{cleanup();worker?.terminate();worker=undefined;reject(error);};
   const abort=()=>fail(new DOMException('Cancelled','AbortError'));
   signal.addEventListener('abort',abort,{once:true});
   worker.onmessage=({data})=>{
    if(data.id!==id)return;
    if(data.type==='result'){cleanup();resolve({text:data.result.lines.map((l:{text:string})=>l.text).join('\n'),lines:data.result.lines.length,elapsedMs:data.result.elapsed_ms,version:'0.11.4',cached:!!data.backend?.result_cached,imageSha256:data.backend?.image_sha256});}
    if(data.type==='error')fail(Error(data.message));
   };
   worker.onerror=e=>fail(Error(e.message||'图片处理失败'));
   worker.postMessage({id,bytes},[bytes]);
  });
 });
 queue=task;
 return task;
}
