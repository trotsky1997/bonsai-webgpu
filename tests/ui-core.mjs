import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import init,{ui_core} from '../web/pkg/bonsai_webgpu.js';
import {fixtures} from './ui-fixtures.mjs';
await init({module_or_path:await readFile(new URL('../web/pkg/bonsai_webgpu_bg.wasm',import.meta.url))});
const call=r=>JSON.parse(ui_core(JSON.stringify(r)));
for(const definition of fixtures){const result=call({op:'validate',definition});assert.equal(result.definition.type,definition.type);assert(result.fields.length);}
const definition=fixtures[4],initial=call({op:'validate',definition}).state;
assert.equal(call({op:'evaluate',definition,state:initial}).outputs[0].value,1200);
assert.equal(call({op:'evaluate',definition,state:{...initial,'param:monthly':200}}).chart.series[0].values[11],2400);
assert.throws(()=>call({op:'validate',definition:{...definition,version:9}}),/version/);
assert.throws(()=>call({op:'merge',definition,base:initial,current:initial,patch:{'param:monthly':-1}}),/Invalid/);
assert.throws(()=>call({op:'merge',definition,base:initial,current:initial,patch:JSON.parse('{"__proto__":{}}')}));
console.log('WASM UI core: six definitions, calculations and bounds passed');
