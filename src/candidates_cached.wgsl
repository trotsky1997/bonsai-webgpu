@group(0) @binding(0) var<storage,read> logits:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<vec2<u32>>;
@group(0) @binding(2) var<uniform> p:vec4<u32>;
var<workgroup> values:array<f32,128>;
var<workgroup> tokens:array<u32,128>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) id:vec3<u32>){
 let size=(p.x+63u)/64u;let start=id.x*size;let end=min(start+size,p.x);
 var local:array<f32,19>;var best=-3.402823466e38;var token=0xffffffffu;var best_index=0u;
 for(var i=0u;i<19u;i++){let t=start+lane+i*128u;var value=-3.402823466e38;if t<end {value=logits[t];}local[i]=value;if t<end&&(value>best||(value==best&&t>token)){best=value;token=t;best_index=i;}}
 for(var rank=0u;rank<p.y;rank++){
  values[lane]=best;tokens[lane]=token;workgroupBarrier();
  for(var step=64u;step>0u;step/=2u){if lane<step{let b=values[lane+step];let bt=tokens[lane+step];if b>values[lane]||(b==values[lane]&&bt>tokens[lane]){values[lane]=b;tokens[lane]=bt;}}workgroupBarrier();}
  let winner=tokens[0];if lane==0u {output[id.x*p.y+rank]=vec2<u32>(winner,bitcast<u32>(values[0]));}workgroupBarrier();
  if rank+1u<p.y && token==winner {
   local[best_index]=-3.402823466e38;best=-3.402823466e38;token=0xffffffffu;
   for(var i=0u;i<19u;i++){let t=start+lane+i*128u;let value=local[i];if t<end&&(value>best||(value==best&&t>token)){best=value;token=t;best_index=i;}}
  }
 }
}
