// Static application shell. The inference Worker manages large model blobs.
const base=new URL('./',self.location.href);
const version='bonsai-webgpu-shell-0.4.1.2:'+base.pathname;
const shell=['./','web/client.mjs?v=0.4.1.1','web/worker.mjs?v=0.4.1.1','web/chat-template.mjs?v=0.4.1.1','web/vendor/jsonrepair.mjs','web/ngram.mjs?v=0.4.1.1','web/ui-core.mjs?v=0.4.1.1','web/ocr/worker.mjs?v=0.4.1.1','web/ocr/ppocr.mjs?v=0.11.4','web/pkg/bonsai_webgpu.js?v=0.4.1.1','web/pkg/bonsai_webgpu_bg.wasm?v=0.4.1.1',"_app/immutable/assets/bundle.FEmzbMte.css","_app/immutable/bundle.DYWXd7Wz.js","_app/immutable/workers/worker-B5gTTf4d.js",'_app/version.json','model/manifest.json','favicon.svg','favicon-dark.svg','favicon.ico'].map(path=>new URL(path,base).href);
self.addEventListener('install',event=>event.waitUntil(caches.open(version).then(cache=>cache.addAll(shell)).then(()=>self.skipWaiting())));
self.addEventListener('activate',event=>event.waitUntil((async()=>{for(const key of await caches.keys())if(key.startsWith('bonsai-webgpu-shell-')&&key.endsWith(':'+base.pathname)&&key!==version)await caches.delete(key);await self.clients.claim();})()));
self.addEventListener('fetch',event=>{
 const url=new URL(event.request.url);const path=url.pathname.startsWith(base.pathname)?'/'+url.pathname.slice(base.pathname.length):url.pathname;
 if(path.startsWith('/web/ocr/')&&path.endsWith('.wasm'))return;
 if(url.origin!==self.location.origin||event.request.method!=='GET'||path.startsWith('/v1/')||path.startsWith('/benchmark')||path==='/props'||(path.startsWith('/model/')&&path!=='/model/manifest.json'))return;
 // Local development uses network first so worker imports don't mask source fixes.
 const local=['127.0.0.1','localhost','[::1]'].includes(self.location.hostname);
 const immutable=!local&&((path.startsWith('/web/')&&url.searchParams.get('v')==='0.4.1')||path.startsWith('/_app/immutable/'));
 event.respondWith((async()=>{const cache=await caches.open(version);if(immutable){const cached=await cache.match(event.request);if(cached)return cached;}return fetch(event.request).then(response=>{if(response.ok){const copy=response.clone();event.waitUntil(caches.open(version).then(cache=>cache.put(event.request,copy)).catch(()=>{}));}return response;}).catch(async()=>await (await caches.open(version)).match(event.request)||Response.error());})());
});
