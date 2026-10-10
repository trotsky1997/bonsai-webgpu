// Rebuild the pinned Svelte frontend and retain the existing icon/splash assets.
import {readFile,writeFile,readdir,mkdir} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
const repair=spawnSync('node_modules/.bin/esbuild',['node_modules/jsonrepair/lib/esm/regular/jsonrepair.js','--bundle','--format=esm','--minify','--outfile=../web/vendor/jsonrepair.mjs'],{cwd:'ui-source',stdio:'inherit'});if(repair.status!==0)process.exit(repair.status??1);
const assets=new Map();for(const f of await readdir('ui'))if(/\.(png|svg|ico)$/.test(f))assets.set(f,await readFile('ui/'+f));
const built=spawnSync('npm',['exec','vite','build'],{cwd:'ui-source',env:{...process.env,LLAMA_UI_OUT_DIR:'../ui'},stdio:'inherit'});if(built.status!==0)process.exit(built.status??1);
for(const [f,bytes]of assets)await writeFile('ui/'+f,bytes);
// The runtime owns the sole service worker, including the hashed Svelte assets.
const files=(await readdir('ui/_app/immutable',{recursive:true,withFileTypes:true})).filter(e=>e.isFile()).map(e=>(e.parentPath+'/'+e.name).replace(/^ui\//,'')).sort();
await writeFile('ui/sw.js',(await readFile('web/sw.js','utf8')).replace('/* UI_ASSETS */',files.map(f=>JSON.stringify(f)).join(',')));
