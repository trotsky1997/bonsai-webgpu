// Export a self-contained static site. Paths resolve relative to the deployed base.
import {cp,mkdir,writeFile,rm,readFile,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {dirname,join,resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
const root=dirname(fileURLToPath(import.meta.url));
const destination=resolve(process.argv[2]||join(root,'work/site'));
if(destination===root||!destination.startsWith(root+'/'))throw Error('Output must be a child directory of the package');
const manifest=await readFile(join(root,'model/manifest.json'));
const expected=createHash('sha256').update(manifest).digest('hex');
if(!(await readFile(join(root,'web/worker.mjs'),'utf8')).includes("const manifestSha='"+expected+"'"))throw Error('Worker manifest SHA-256 is stale; rerun the model conversion');
await rm(destination,{recursive:true,force:true});await mkdir(destination,{recursive:true});
await cp(join(root,'ui'),destination,{recursive:true});
await cp(join(root,'web'),join(destination,'web'),{recursive:true,filter:path=>!path.endsWith('bench.html')&&!path.endsWith('bench.mjs')});
await cp(join(root,'model'),join(destination,'model'),{recursive:true,filter:path=>!path.endsWith('tokenizer.json')});
const immutable=join(root,'ui/_app/immutable');
const assets=(await readdir(immutable,{recursive:true,withFileTypes:true})).filter(e=>e.isFile()).map(e=>join(e.parentPath,e.name).slice(join(root,'ui').length+1)).sort();
if(assets.length<2)throw Error('Missing UI bundle assets');
await writeFile(join(destination,'sw.js'),(await readFile(join(root,'web/sw.js'),'utf8')).replace('/* UI_ASSETS */',assets.map(a=>JSON.stringify(a)).join(',')));
for(const name of ['LICENSE','MODEL_LICENSE','LLAMA_CPP_LICENSE'])await cp(join(root,name),join(destination,name));
await writeFile(join(destination,'.nojekyll'),'');
await writeFile(join(destination,'deployment.json'),JSON.stringify({version:'0.4.1',commit:process.env.GITHUB_SHA||null,checkpoint:JSON.parse(await readFile(join(root,'model/manifest.json'),'utf8')).checkpoint_sha256})+'\n');
console.log(destination);

for(const name of ['worker.mjs','pkg'])await cp(join(root,'benchmarks/v020',name),join(destination,'benchmarks/v020',name),{recursive:true});
