import {createOcr,supportsRelaxedSimd} from './ppocr.mjs?v=0.11.4';
const hashes={
 'ppocr_rs.wasm':'7bb573390bb771b357a8fd4de06e575f167b70d9ad69c1ee4f60345117727134',
 'ppocr_rs_fma.wasm':'c153e994e1580e3abcec0749ce29f5ec7ade30296124dca135f685eadd26aea3'
};
let ready,busy=false;
async function load(){
 postMessage({type:'loading',message:'正在准备图片识字模型（首次约 12.5 MB）…'});
 const file=supportsRelaxedSimd()?'ppocr_rs_fma.wasm':'ppocr_rs.wasm',hash=hashes[file];
 const url=new URL(file,import.meta.url);url.searchParams.set('sha256',hash);
 const cache=await caches.open('bonsai-ocr-v0.11.4');let response=await cache.match(url),cached=!!response;
 if(!response){response=await fetch(url);if(!response.ok)throw Error('OCR 模型下载失败：'+response.status);}
 const bytes=await response.arrayBuffer();
 if(!cached){const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');if(digest!==hash)throw Error('OCR 模型校验失败，请重试');await cache.put(url,new Response(bytes,{headers:{'Content-Type':'application/wasm'}}));}
 const model=await createOcr({wasmBytes:bytes,precision:'int8',detMaxSide:960,recMinWidth:320});
 return {model,file,cached};
}
onmessage=async({data})=>{
 if(busy)return postMessage({type:'error',id:data.id,message:'OCR 正在识别另一张图片'});
 busy=true;
 try{
  const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data.bytes)),x=>x.toString(16).padStart(2,'0')).join('');
  const resultCache=await caches.open('bonsai-ocr-results-v0.11.4-int8-960-320');
  const key=new URL('./results/'+digest,import.meta.url).href;
  const hit=await resultCache.match(key);
  if(hit){try{const result=await hit.json();if(Array.isArray(result.lines)&&result.width>0&&result.height>0){postMessage({type:'result',id:data.id,result,backend:{version:'0.11.4',result_cached:true,image_sha256:digest}});return;}}catch{await resultCache.delete(key);}}
  const {model,file,cached}=await(ready??=load().catch(e=>{ready=undefined;throw e;}));
  postMessage({type:'loading',id:data.id,message:'正在本地识别…'});
  const result=model.ocr(data.bytes);
  await resultCache.put(key,new Response(JSON.stringify(result),{headers:{'Content-Type':'application/json'}}));
  const entries=await resultCache.keys();for(const old of entries.slice(0,Math.max(0,entries.length-128)))await resultCache.delete(old);
  postMessage({type:'result',id:data.id,result,backend:{version:'0.11.4',file,cached,result_cached:false,image_sha256:digest,info:model.info}});
 }catch(e){postMessage({type:'error',id:data.id,message:String(e?.message??e)});}finally{busy=false;}
};
