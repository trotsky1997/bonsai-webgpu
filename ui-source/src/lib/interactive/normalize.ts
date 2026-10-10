// Small checkpoints sometimes copy JSON Schema's `type: object` into a call.
// Repair only unambiguous structural mistakes; do not invent values or fields.
export function normalizeUiArgs(input:any){
 const args=structuredClone(input),data=args?.data;
 if(!data||typeof data!=='object'||Array.isArray(data))return args;
 if(args.type==='object'){
  const shapes=[
   ['parameters',Array.isArray(data.controls)&&Array.isArray(data.outputs)],
   ['comparison',Array.isArray(data.rows)&&Array.isArray(data.columns)],
   ['diagram',Array.isArray(data.nodes)&&Array.isArray(data.edges)],
   ['chart',['line','bar','pie'].includes(data.kind)&&Array.isArray(data.labels)&&Array.isArray(data.series)]
  ].filter(([,matches])=>matches);
  if(shapes.length===1)args.type=shapes[0][0];
 }
 if(args.type==='parameters'&&Array.isArray(data.outputs))for(const output of data.outputs){
  if(output&&typeof output.expr==='string'&&Array.isArray(output.args)){
   output.expr={op:output.expr,args:output.args};delete output.args;
  }
 }
 return args;
}
