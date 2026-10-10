import {readFileSync,unlinkSync} from 'node:fs';import {dirname,join} from 'node:path';import {fileURLToPath} from 'node:url';
const file=join(dirname(fileURLToPath(import.meta.url)),'.runtime/server.pid');
try{const pid=Number(readFileSync(file,'utf8'));process.kill(pid,'SIGTERM');unlinkSync(file);console.log('静态服务已停止。');}catch{console.log('静态服务未运行。');}
