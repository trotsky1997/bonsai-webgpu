// Static file host + local benchmark result collector. No model runtime.
import http from 'node:http';
import {createReadStream,statSync,readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {dirname,resolve,extname,join} from 'node:path';
import {fileURLToPath} from 'node:url';
const root=dirname(fileURLToPath(import.meta.url)),port=Number(process.env.BONSAI_WEBGPU_PORT||18902);
const types={'.html':'text/html; charset=utf-8','.js':'text/javascript','.mjs':'text/javascript','.wasm':'application/wasm','.json':'application/json','.bin':'application/octet-stream','.css':'text/css','.png':'image/png','.jpg':'image/jpeg','.svg':'image/svg+xml','.woff2':'font/woff2','.ico':'image/x-icon'};
const jobs=[],results=new Map();
const server=http.createServer(async(req,res)=>{
 try{
  const allowed=[`127.0.0.1:${port}`,`localhost:${port}`];if(!allowed.includes(req.headers.host))throw Error('Invalid host');
  if(req.headers.origin&&!allowed.map(h=>'http://'+h).includes(req.headers.origin))throw Error('Invalid origin');
  const path=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
  if(path==='/benchmark-next'&&req.method==='GET'){res.writeHead(200,{'Content-Type':'application/json','Cache-Control':'no-store'});return res.end(JSON.stringify((()=>{const variant=new URL(req.url,'http://localhost').searchParams.get('variant')||'optimized';const i=jobs.findIndex(j=>(j.variant||'optimized')===variant);return i<0?null:jobs.splice(i,1)[0];})()));}
  if(path==='/benchmark-status'&&req.method==='GET'){res.writeHead(200,{'Content-Type':'application/json','Cache-Control':'no-store'});return res.end(JSON.stringify(Object.fromEntries(results)));}
  if(path==='/benchmark-command'&&req.method==='POST'){let text='';for await(const chunk of req)text+=chunk;if(text.length>65536)throw Error('Too large');jobs.push(JSON.parse(text));res.writeHead(200,{'Content-Type':'application/json'});return res.end('{"queued":true}');}
  if(path==='/benchmark-result'&&req.method==='POST'){
   let body='',size=0;for await(const chunk of req){size+=chunk.length;if(size>8*1024*1024)throw Error('Too large');body+=chunk;}
   const value=JSON.parse(body);if(value.job_id)results.set(value.job_id,value);mkdirSync(join(root,'benchmarks'),{recursive:true});writeFileSync(join(root,'benchmarks/browser-latest.json'),JSON.stringify(value,null,2)+'\n');
   res.writeHead(200,{'Content-Type':'application/json'});return res.end('{"saved":true}');
  }
  if(!['GET','HEAD'].includes(req.method)){res.writeHead(405);return res.end('Static host: inference runs in your browser.');}
  if(path.startsWith('/v1/')||path==='/props'){res.writeHead(410);return res.end('Inference API exists only inside the browser Worker.');}
  let file;
  if(path.startsWith('/model/')||path.startsWith('/web/')||path.startsWith('/benchmarks/'))file=resolve(root,'.'+path);
  else if(path==='/bench.html')file=join(root,'web/bench.html');
  else file=resolve(root,'ui',path==='/'?'index.html':'.'+path);
  if(!file.startsWith(root+'/')||path.includes('..'))throw Error('Invalid path');
  if(path==='/sw.js')file=join(root,'web/sw.js');
  const stat=statSync(file);if(!stat.isFile())throw Error('Not a file');
  res.writeHead(200,{'Content-Type':types[extname(file)]||'application/octet-stream','Content-Length':stat.size,'Cache-Control':path.startsWith('/model/')?'public, max-age=3600':'no-store','Cross-Origin-Opener-Policy':'same-origin','Cross-Origin-Embedder-Policy':'require-corp'});
  if(req.method==='HEAD')return res.end();createReadStream(file).pipe(res);
 }catch(e){res.writeHead(404);res.end(String(e.message));}
});
server.listen(port,'127.0.0.1',()=>console.log(JSON.stringify({event:'listening',url:`http://127.0.0.1:${port}`,pid:process.pid,backend:'static-only'})));
