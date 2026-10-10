// Static application shell. The inference Worker manages large model blobs.
const base=new URL('./',self.location.href);
const version='bonsai-webgpu-shell-0.2.0:'+base.pathname;
const shell=['./','web/client.mjs','web/worker.mjs','web/pkg/bonsai_webgpu.js','web/pkg/bonsai_webgpu_bg.wasm','_app/immutable/bundle.BQv1Gmi9.js','_app/immutable/assets/bundle.BHySb6NS.css','_app/version.json','model/manifest.json','favicon.svg','favicon-dark.svg','favicon.ico'].map(path=>new URL(path,base).href);
self.addEventListener('install',event=>event.waitUntil(caches.open(version).then(cache=>cache.addAll(shell)).then(()=>self.skipWaiting())));
self.addEventListener('activate',event=>event.waitUntil(self.clients.claim()));
self.addEventListener('fetch',event=>{
 const url=new URL(event.request.url);const path=url.pathname.startsWith(base.pathname)?'/'+url.pathname.slice(base.pathname.length):url.pathname;
 if(url.origin!==self.location.origin||event.request.method!=='GET'||path.startsWith('/v1/')||path.startsWith('/benchmark')||path==='/props'||(path.startsWith('/model/')&&path!=='/model/manifest.json'))return;
 event.respondWith(fetch(event.request).then(response=>{if(response.ok){const copy=response.clone();event.waitUntil(caches.open(version).then(cache=>cache.put(event.request,copy)).catch(()=>{}));}return response;}).catch(async()=>await (await caches.open(version)).match(event.request)||Response.error()));
});
