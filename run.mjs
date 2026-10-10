import {spawn} from 'node:child_process';
import {dirname,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {mkdirSync,openSync,writeFileSync,readFileSync} from 'node:fs';
const root=dirname(fileURLToPath(import.meta.url)),runtime=join(root,'.runtime');mkdirSync(runtime,{recursive:true});
try{const pid=Number(readFileSync(join(runtime,'server.pid'),'utf8'));process.kill(pid,0);console.log(`已运行：http://127.0.0.1:${process.env.BONSAI_WEBGPU_PORT||18902}/`);process.exit(0);}catch{}
const log=openSync(join(runtime,'server.log'),'a');
const child=spawn(process.execPath,[join(root,'server.mjs')],{cwd:root,env:process.env,detached:true,stdio:['ignore',log,log]});
writeFileSync(join(runtime,'server.pid'),String(child.pid));child.unref();
console.log(`聊天：http://127.0.0.1:${process.env.BONSAI_WEBGPU_PORT||18902}/`);
