import {expect,it,vi,afterEach} from 'vitest';
afterEach(()=>vi.unstubAllGlobals());
it('serializes local image jobs, cancels queued work and reuses the worker',async()=>{
 const workers:any[]=[];const messages:any[]=[];
 class TestWorker {onmessage:any;onerror:any;terminate=vi.fn();constructor(){workers.push(this);}postMessage(message:any){messages.push(message);queueMicrotask(()=>this.onmessage({data:{id:message.id,type:'result',result:{lines:[{text:'中文2026'}],elapsed_ms:25}}}));}}
 vi.stubGlobal('Worker',TestWorker);vi.stubGlobal('document',{baseURI:'http://local/app/'});
 vi.stubGlobal('fetch',vi.fn(async()=>new Response(new Uint8Array([1,2,3]))));
 const {recognizeImage}=await import('$lib/ocr/client');const a=new AbortController(),b=new AbortController(),c=new AbortController();b.abort();
 const results=await Promise.allSettled([recognizeImage('data:image/png;base64,AQID',a.signal),recognizeImage('data:image/png;base64,AQID',b.signal),recognizeImage('data:image/png;base64,AQID',c.signal)]);
 expect(results[0]).toMatchObject({status:'fulfilled',value:{text:'中文2026',lines:1,version:'0.11.4'}});expect(results[1].status).toBe('rejected');expect(results[2].status).toBe('fulfilled');expect(workers).toHaveLength(1);expect(messages).toHaveLength(2);
});
