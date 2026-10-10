// Port of this checkpoint's chat_template.jinja, including Qwen tool messages.
// This module has no GPU or browser dependencies, so protocol fixtures can run in Node.
function contentOf(message){
 if(Array.isArray(message.content))return message.content.map(p=>{if(p.type!=='text')throw Error('只支持文本消息');return p.text;}).join('');
 return message.content??'';
}
export function toolsFor(body){
 if(body.tool_choice!==undefined&&!['auto','none'].includes(body.tool_choice))throw Error('当前工具选择支持 auto 或 none');
 if(body.tool_choice==='none')return [];
 if(body.tools!==undefined&&!Array.isArray(body.tools))throw Error('tools must be an array');
 const tools=body.tools??[];const names=new Set();
 for(const tool of tools){
  const name=tool?.function?.name;
  if(tool.type!=='function'||typeof name!=='string'||!/^[-\w.]{1,128}$/.test(name)||names.has(name))throw Error('Invalid or duplicate tool definition');
  names.add(name);
 }
 return tools;
}
export function renderPrompt(body){
 if(!Array.isArray(body.messages)||!body.messages.length)throw Error('messages required');
 const messages=body.messages.map(m=>({...m,role:m.role==='developer'?'system':m.role}));
 const tools=toolsFor(body);let prompt='';const lastUser=messages.findLastIndex(m=>m.role==='user'&&!(contentOf(m).startsWith('<tool_response>')&&contentOf(m).endsWith('</tool_response>')));
 if(tools.length){
  prompt='<|im_start|>system\n'+(messages[0].role==='system'?contentOf(messages[0])+'\n\n':'');
  prompt+='# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>';
  for(const tool of tools)prompt+='\n'+JSON.stringify(tool);
  prompt+='\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n';
 }
 for(const [index,message] of messages.entries()){
  const role=message.role;if(!['system','user','assistant','tool'].includes(role))throw Error('Unsupported message role');
  if(role==='system'&&index===0&&tools.length)continue;
  let content=contentOf(message),reasoning=message.reasoning_content||'';
  if(role==='tool'){
   if(index===0||messages[index-1].role!=='tool')prompt+='<|im_start|>user';
   prompt+='\n<tool_response>\n'+content+'\n</tool_response>';
   if(index===messages.length-1||messages[index+1].role!=='tool')prompt+='<|im_end|>\n';
   continue;
  }
  if(role==='assistant'&&content.includes('</think>')){if(!reasoning)reasoning=content.split('</think>')[0].split('<think>').at(-1).trim();content=content.split('</think>').at(-1).replace(/^\n+/,'');}
  prompt+=`<|im_start|>${role}\n`;
  if(role==='assistant'&&index>lastUser&&(index===messages.length-1||reasoning))prompt+=`<think>\n${reasoning.trim()}\n</think>\n\n`;
  prompt+=content;
  for(const [i,call] of (message.tool_calls??[]).entries()){
   const fn=call.function??call;
   const args=typeof fn.arguments==='string'?fn.arguments:JSON.stringify(fn.arguments??{});
   // Validate assistant history arguments before rendering the protocol.
   JSON.parse(args);
   if(i||content)prompt+='\n';
   prompt+='<tool_call>\n{"name": '+JSON.stringify(fn.name)+', "arguments": '+args+'}\n</tool_call>';
  }
  prompt+='<|im_end|>\n';
 }
 if(body.continue_final_message){if(messages.at(-1).role!=='assistant')throw Error('Continuation needs assistant');return prompt.slice(0,-'<|im_end|>\n'.length);}
 return prompt+'<|im_start|>assistant\n<think>\n\n</think>\n\n';
}
const open='<tool_call>',close='</tool_call>';
export class ToolCallParser{
 constructor(tools,onText){this.allowed=new Set(tools.map(t=>t.function.name));this.onText=onText;this.pending='';this.inside=false;this.calls=[];}
 push(text){
  this.pending+=text;
  for(;;){
   const tag=this.inside?close:open,index=this.pending.indexOf(tag);
   if(index<0){
    if(!this.inside){let hold=0;for(let n=1;n<tag.length;n++)if(this.pending.endsWith(tag.slice(0,n)))hold=n;this.onText(this.pending.slice(0,this.pending.length-hold));this.pending=this.pending.slice(this.pending.length-hold);}
    return;
   }
   const before=this.pending.slice(0,index);this.pending=this.pending.slice(index+tag.length);
   if(!this.inside){this.onText(before);this.inside=true;}
   else{
    let value;try{value=JSON.parse(before);}catch{throw Error('模型生成了无效的工具调用 JSON');}
    const args=value?.arguments;
    if(!this.allowed.has(value?.name))throw Error('模型调用了未启用的工具: '+value?.name);
    if(!args||typeof args!=='object'||Array.isArray(args))throw Error('工具 arguments 必须是 JSON 对象');
    if(this.calls.length>=8)throw Error('单次回复最多允许 8 个工具调用');
    this.calls.push({type:'function',function:{name:value.name,arguments:JSON.stringify(args)}});this.inside=false;
   }
  }
 }
 finish(cancelled=false){
  if(cancelled){this.calls=[];this.pending='';return [];}
  if(this.inside||this.pending.startsWith('<tool_'))throw Error('工具调用被截断，请增加回复 token 上限或重试');
  this.onText(this.pending);this.pending='';return this.calls;
 }
}
