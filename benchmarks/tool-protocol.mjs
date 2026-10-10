import assert from 'node:assert/strict';
import {renderPrompt,ToolCallParser,toolsFor} from '../web/chat-template.mjs';
const tools=[{type:'function',function:{name:'get_datetime',parameters:{type:'object',properties:{}}}}];
const message={role:'user',content:'今天是几号？'};
const original='<|im_start|>user\n今天是几号？<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n';
assert.equal(renderPrompt({messages:[message]}),original);
assert.equal(renderPrompt({messages:[message],tools,tool_choice:'none'}),original);
assert.deepEqual(toolsFor({tools,tool_choice:'none'}),[]);
const p=renderPrompt({tools,messages:[{role:'system',content:'Test'},message,{role:'assistant',content:null,tool_calls:[{function:{name:'get_datetime',arguments:'{}'}}]},{role:'tool',content:'{"date":"2026-10-10"}'},{role:'tool',content:'second'}]});
assert.ok(p.includes('<tools>\n'+JSON.stringify(tools[0])+'\n</tools>'));
assert.ok(p.includes('<tool_call>\n{"name": "get_datetime", "arguments": {}}\n</tool_call>'));
assert.ok(p.includes('<|im_start|>user\n<tool_response>\n{"date":"2026-10-10"}\n</tool_response>\n<tool_response>\nsecond\n</tool_response><|im_end|>'));
const text='先查时间。<tool_call>\n{"name":"get_datetime","arguments":{}}\n</tool_call>\n';let splitChecks=0;
for(let split=0;split<=text.length;split++){
 let content='';const parser=new ToolCallParser(tools,t=>content+=t);parser.push(text.slice(0,split));parser.push(text.slice(split));
 const calls=parser.finish();assert.equal(content,'先查时间。\n');assert.deepEqual(calls,[{type:'function',function:{name:'get_datetime',arguments:'{}'}}]);splitChecks++;
}
let streamed='';const charParser=new ToolCallParser(tools,t=>streamed+=t);for(const char of text)charParser.push(char);assert.equal(charParser.finish().length,1);assert.equal(streamed,'先查时间。\n');
for(const xml of ['<tool_call>{bad}</tool_call>','<tool_call>{"name":"unknown","arguments":{}}</tool_call>','<tool_call>{"name":"get_datetime","arguments":[]}</tool_call>','<tool_call>{','<tool_cal']){
 assert.throws(()=>{const p=new ToolCallParser(tools,()=>{});p.push(xml);p.finish();});
}
const cancelled=new ToolCallParser(tools,()=>{});cancelled.push('<tool_call>{');assert.deepEqual(cancelled.finish(true),[]);
console.log(JSON.stringify({plain_prompt_exact:true,tool_choice_none:true,tool_response_grouping:true,split_boundaries:splitChecks,character_stream:true,invalid_or_truncated_rejected:5,cancelled_no_execution:true}));
