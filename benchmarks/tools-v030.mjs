// Run inside the chat page so fetch uses the actual browser transport.
export async function checkTools(){
 const tools=[{type:'function',function:{name:'get_datetime',description:'Returns the current local date and time in ISO 8601 format, with the IANA time zone name',parameters:{type:'object',properties:{},required:[]}}}];
 const results=[];
 const complete=async(messages,stream)=>{
  const response=await fetch('./v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages,tools,temperature:0,max_tokens:128,stream})});
  if(!stream){const data=await response.json();if(data.error)throw Error(data.error.message);return data;}
  const frames=(await response.text()).split('\n').filter(line=>line.startsWith('data: {')).map(line=>JSON.parse(line.slice(6)));
  if(frames.some(f=>f.error))throw Error(JSON.stringify(frames.find(f=>f.error)));
  let content='',tool_calls=[],finish_reason;
  for(const frame of frames){const choice=frame.choices?.[0];content+=choice?.delta?.content??'';if(choice?.delta?.tool_calls)tool_calls.push(...choice.delta.tool_calls);if(choice?.finish_reason)finish_reason=choice.finish_reason;}
  return{choices:[{message:{role:'assistant',content:content||null,...(tool_calls.length?{tool_calls:tool_calls.map(({index,...call})=>call)}:{})},finish_reason}]};
 };
 for(const [question,stream]of [['今天是几号？',true],['What is the current date and time?',false]]){
  const messages=[{role:'user',content:question}],first=await complete(messages,stream);
  const call=first.choices[0].message.tool_calls?.[0];if(call?.function.name!=='get_datetime'||first.choices[0].finish_reason!=='tool_calls')throw Error('No real date/time call: '+JSON.stringify(first));
  const now=new Date(),pad=n=>String(n).padStart(2,'0'),offset=-now.getTimezoneOffset(),sign=offset>=0?'+':'-';
  const result={result:`${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())}T${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}${sign}${pad(Math.floor(Math.abs(offset)/60))}:${pad(Math.abs(offset)%60)}`,timezone:Intl.DateTimeFormat().resolvedOptions().timeZone};
  messages.push(first.choices[0].message,{role:'tool',tool_call_id:call.id,content:JSON.stringify(result)});
  const second=await complete(messages,stream);
  if(second.choices[0].message.tool_calls?.length)throw Error('Unnecessary repeated time call');
  results.push({question,stream,first,actual_tool_result:result,second});
 }
 const ordinary=await complete([{role:'user',content:'Say hello in one short sentence.'}],true);
 if(ordinary.choices[0].message.tool_calls?.length)throw Error('Unnecessary tool use for greeting');
 return{results,ordinary};
}
