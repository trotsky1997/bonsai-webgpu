import 'fake-indexeddb/auto';
import {beforeAll,expect,it,vi} from 'vitest';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import init,{ui_core} from '../../../web/pkg/bonsai_webgpu.js';
import {fixtures} from '../../../tests/ui-fixtures.mjs';
vi.mock('$lib/interactive/core',()=>({core:async()=>({call:(r:unknown)=>JSON.parse(ui_core(JSON.stringify(r)))})}));
import {interactiveStore as store} from '$lib/interactive/store.svelte';
beforeAll(async()=>{
 await init({module_or_path:await readFile(resolve('../web/pkg/bonsai_webgpu_bg.wasm'))});
 vi.stubGlobal('document',{baseURI:'http://localhost/test/'});
 vi.stubGlobal('sessionStorage',{getItem:()=>null,setItem:()=>{},removeItem:()=>{}});
 vi.stubGlobal('window',{dispatchEvent:()=>{}});
});
it('creates immutable definitions, replays calls once and returns correctable errors',async()=>{
 const args=fixtures[4];const a=await store.present(args,'c1','call1');const again=await store.present(args,'c1','call1');
 expect(again.version).toBe(1);expect(again.calls).toEqual(['call1']);
 await expect(store.present({...args,title:'changed'},'c1','call1')).rejects.toThrow(/不能修改/);
 const error=await store.execute('present_ui',{...args,version:2},'c1','bad');expect(error.isError).toBe(true);expect(error.content).toContain('Unsupported');
 expect(a.state['param:monthly']).toBe(100);
});
it('serializes simultaneous IDB writes; merges independent fields; conflicts atomically',async()=>{
 const a=await store.present(fixtures[4],'c2','call2');
 const [first,second]=await Promise.all([store.patch(a.key,1,{'param:monthly':200},'a'),store.patch(a.key,1,{'param:enabled':false},'b')]);
 expect(first.status).toBe('ok');expect(second.status).toBe('ok');expect(second.record.version).toBe(3);
 const conflict=await store.patch(a.key,1,{'param:monthly':300,'param:mode':'b'},'c');expect(conflict.status).toBe('conflict');expect(conflict.record.state['param:mode']).toBe('a');
 const retry=await store.patch(a.key,3,{'param:monthly':300},'d');expect(retry.record.version).toBe(4);
 const replay=await store.patch(a.key,3,{'param:monthly':999},'d');expect(replay.record.version).toBe(4);expect(replay.record.state['param:monthly']).toBe(300);
});
it('binds an immutable version snapshot after flushing, scoped to conversation',async()=>{
 const a=await store.present(fixtures[4],'c3','call3');
 store.register(a.key,async()=>{await store.patch(a.key,1,{'param:monthly':200},'flush')});
 await store.intent(a,'Continue');
 const accepted=await store.bind('c3');expect(accepted).toContain('"version":2');expect(accepted).toContain('2400');
 await store.patch(a.key,2,{'param:monthly':300},'later');expect(accepted).toContain('"param:monthly":200');expect(await store.bind('other')).toBe('');
 store.acknowledge('c3');expect(await store.bind('c3')).toBe('');
 await store.deleteConversation('c3');expect(await store.read(a.key)).toBeUndefined();expect(await store.read('c2/budget')).toBeDefined();
});
it('removes orphan components after message deletion while retaining live call references',async()=>{
 const a=await store.present(fixtures[0],'delete-test','old');await store.present(fixtures[0],'delete-test','live');
 await store.retainCalls('delete-test',new Set(['live']));expect((await store.read(a.key))?.calls).toEqual(['live']);
 await store.retainCalls('delete-test',new Set());expect(await store.read(a.key)).toBeUndefined();
});

it('repairs unambiguous schema-object type and flattened output expressions without inventing values',async()=>{
 const input=structuredClone(fixtures[4]);input.id='normalized';(input as any).type='object';(input.data as any).outputs=[{id:'annual',label:'Total',expr:'mul',args:[{param:'monthly'},12]}];
 const record=await store.present(input,'repair','repaired-call');expect(record.definition.type).toBe('parameters');
 expect((await store.calculate(record.definition,record.state)).outputs[0].value).toBe(1200);
});
