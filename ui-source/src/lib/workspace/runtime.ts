import {minimatch} from 'minimatch';
import {Bash,InMemoryFs} from 'just-bash/browser';
export interface Entry {path:string;type:'file'|'directory'|'symlink';content?:Uint8Array;target?:string;mode:number;mtime:number}
export interface Snapshot {version:number;entries:Entry[];imports?:string[];operations:{id:string;result:string;isError?:boolean}[]}
export const LIMITS={maxExecutionTimeMs:10000,maxCommandCount:3000,maxFileSystemBytes:16*1024**2,maxOutputSize:262144,maxSourceBytes:131072};
const string=(v:unknown,name:string)=>{if(typeof v!=='string')throw Error(name+' must be a string');return v;};
export async function createWorkspace(snapshot?:Snapshot){
 const fs=new InMemoryFs({}, {maxTotalBytes:LIMITS.maxFileSystemBytes});
 for(const e of snapshot?.entries??[]){if(e.type==='directory')await fs.mkdir(e.path,{recursive:true});else if(e.type==='file')await fs.writeFile(e.path,e.content??new Uint8Array());else await fs.symlink(e.target??'',e.path);await fs.chmod(e.path,e.mode);await fs.utimes(e.path,new Date(e.mtime),new Date(e.mtime));}
 await fs.mkdir('/workspace',{recursive:true});
 const bash=new Bash({fs,cwd:'/workspace',executionLimitProfile:'hardened',executionLimits:LIMITS});
 const path=(input:unknown)=>fs.resolvePath('/workspace',string(input,'path'));
 async function serialize():Promise<Entry[]>{const paths=fs.getAllPaths().sort();if(paths.length>4096)throw Error('Workspace entry limit exceeded (4096)');const out:Entry[]=[];let bytes=0;for(const p of paths){const s=await fs.lstat(p);if(s.isFile){bytes+=s.size;if(bytes>LIMITS.maxFileSystemBytes)throw Error('Serialized workspace exceeds 16 MiB');}out.push({path:p,type:s.isDirectory?'directory':s.isSymbolicLink?'symlink':'file',mode:s.mode,mtime:s.mtime.getTime(),...(s.isFile?{content:await fs.readFileBuffer(p)}:s.isSymbolicLink?{target:await fs.readlink(p)}:{})});}return out;}
 async function execute(name:string,args:Record<string,unknown>,signal?:AbortSignal){
  if(args.__protocol_error)throw Error(String(args.__protocol_error));
  let result:string;
  if(name==='bash'){const r=await bash.exec(string(args.command,'command'),{signal});result=r.stdout+(r.stderr?'\nSTDERR:\n'+r.stderr:'')+'\nExit code: '+r.exitCode;}
  else if(name==='read'){const text=await fs.readFile(path(args.path));const offset=args.offset===undefined?1:Number(args.offset),limit=args.limit===undefined?100:Number(args.limit);if(!Number.isInteger(offset)||offset<1||!Number.isInteger(limit)||limit<1||limit>500)throw Error('offset>=1 and limit 1..500');result=text.split('\n').slice(offset-1,offset-1+limit).join('\n');}
  else if(name==='glob'||name==='grep'){
   const pattern=string(args.pattern,'pattern');if(pattern.length>512)throw Error('pattern exceeds 512 characters');
   const root=path(args.path??'/workspace'),limit=Number(args.limit??(name==='glob'?200:100));
   if(!Number.isInteger(limit)||limit<1||limit>(name==='glob'?1000:200))throw Error('Invalid result limit');
   if(args.glob!==undefined&&(typeof args.glob!=='string'||args.glob.length>512))throw Error('Invalid glob filter');
   const candidates=fs.getAllPaths().sort().filter(p=>p===root||p.startsWith(root.replace(/\/$/,'')+'/'));
   const files:string[]=[];for(const p of candidates){const stat=await fs.lstat(p);if(stat.isFile||stat.isSymbolicLink)files.push(p);}
   const match=(p:string,glob:string)=>minimatch(glob.startsWith('/')?p:(p===root?p.split('/').at(-1)!:p.slice(root.replace(/\/$/,'').length+1)),glob,{dot:true,noext:true,nobrace:true,nonegate:true,matchBase:!glob.includes('/')});
   if(name==='glob'){const found=files.filter(p=>match(p,pattern));result=found.slice(0,limit).join('\n')+(found.length>limit?'\n[More paths omitted; narrow the pattern.]':'');if(!result)result='No matching files';}
   else {
    const selected=files.filter(p=>args.glob===undefined||match(p,String(args.glob)));if(selected.length>1000)throw Error('Too many files; narrow path or glob');
    if(!selected.length)result='No matches';else{
     const quote=(v:string)=>"'"+v.replace(/'/g,"'\"'\"'")+"'";
     const r=await bash.exec('grep -n '+(args.literal===true?'-F':'-E')+(args.ignore_case===true?' -i':'')+' -- '+quote(pattern)+' '+selected.map(quote).join(' '),{signal});
     if(r.exitCode>1||r.stderr)throw Error(r.stderr||'grep failed');const lines=r.stdout.trimEnd().split('\n').map(line=>selected.length===1?selected[0]+':'+line:line);result=r.exitCode===1?'No matches':lines.slice(0,limit).join('\n')+(lines.length>limit?'\n[More matches omitted; narrow pattern or path.]':'');
    }
   }
  }
  else if(name==='write'||name==='edit'){
   const p=path(args.path);let content:string;
   if(name==='write')content=string(args.content,'content');else{const before=await fs.readFile(p),old=string(args.oldText,'oldText');if(!old||before.split(old).length!==2)throw Error('oldText must match exactly once. Read the file and retry.');content=before.replace(old,string(args.newText,'newText'));}
   if(new TextEncoder().encode(content).length>1024**2)throw Error('File writes are limited to 1 MiB');
   await fs.mkdir(p.slice(0,p.lastIndexOf('/'))||'/',{recursive:true});await fs.writeFile(p,content);result=(name==='write'?'Wrote ':'Edited ')+p;
  }else throw Error('Unknown workspace operation');
  return result.length>4000?result.slice(0,4000)+'\n[Output truncated; use read with offset/limit or shell filters.]':result;
 }
 return {fs,execute,serialize};
}
