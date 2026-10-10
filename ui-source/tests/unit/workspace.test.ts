import {expect,it} from 'vitest';
import {createWorkspace,LIMITS} from '$lib/workspace/runtime';
it('executes shell and file tools, restores directories/symlinks, and rejects ambiguous edits',async()=>{
 const w=await createWorkspace();await w.execute('write',{path:'sales.csv',content:'units,price\n2,3\n4,5\n'});
 expect(await w.execute('bash',{command:"awk -F, 'NR>1 {s+=$1*$2} END {print s}' sales.csv"})).toContain('26');
 await w.execute('write',{path:'index.html',content:'<h1>Hello</h1>'});await w.execute('edit',{path:'index.html',oldText:'Hello',newText:'世界'});await expect(w.execute('edit',{path:'index.html',oldText:'missing',newText:'x'})).rejects.toThrow(/exactly once/);
 await w.execute('bash',{command:'mkdir -p nested; ln -s ../index.html nested/link'});
 const copy=await createWorkspace({version:1,entries:await w.serialize(),operations:[]});expect(await copy.execute('read',{path:'nested/link'})).toBe('<h1>世界</h1>');
});
it('has no host runtimes or network tools and bounds loops, output, source and files',async()=>{
 const w=await createWorkspace();for(const command of ['node -e 1','python3 -c 1','curl https://example.com','npm install'])expect(await w.execute('bash',{command})).toContain('127');
 expect(await w.execute('bash',{command:'while true; do :; done'})).toMatch(/limit|exceeded|Exit code: [^0]/i);
 expect(await w.execute('bash',{command:' '.repeat(LIMITS.maxSourceBytes+1)})).toContain('script input size limit exceeded');
 await expect(w.execute('write',{path:'large',content:'a'.repeat(1024**2+1)})).rejects.toThrow(/1 MiB/);
 expect((await w.execute('bash',{command:"printf '%6000s' x"})).length).toBeLessThan(4200);
});

it('glob and grep search recursively with line numbers, filters, bounds and literal-safe arguments',async()=>{
 const w=await createWorkspace();await w.execute('write',{path:'src/a.rs',content:'Alpha\nBeta\nalpha\n'});await w.execute('write',{path:'note.txt',content:'Alpha\n'});
 expect(await w.execute('glob',{pattern:'**/*.rs'})).toBe('/workspace/src/a.rs');
 expect(await w.execute('grep',{pattern:'alpha',ignore_case:true,glob:'**/*.rs'})).toBe('/workspace/src/a.rs:1:Alpha\n/workspace/src/a.rs:3:alpha');
 expect(await w.execute('grep',{pattern:'Alpha',limit:1})).toContain('More matches omitted');
 expect(await w.execute('grep',{pattern:"'; touch injected; #",literal:true})).toBe('No matches');
 expect(await w.execute('glob',{pattern:'**/injected'})).toBe('No matching files');
 expect(await w.execute('glob',{pattern:'missing/**'})).toBe('No matching files');
 await expect(w.execute('grep',{pattern:'[',literal:false})).rejects.toThrow();
 await expect(w.execute('glob',{pattern:'*',limit:1001})).rejects.toThrow();
});
