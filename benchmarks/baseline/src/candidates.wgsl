@group(0) @binding(0) var<storage,read> logits:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<vec2<u32>>;
@group(0) @binding(2) var<uniform> p:vec4<u32>;
var<workgroup> values:array<f32,128>;
var<workgroup> tokens:array<u32,128>;
var<workgroup> removed:array<u32,64>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) id:vec3<u32>){
 let size=(p.x+63u)/64u;let start=id.x*size;let end=min(start+size,p.x);
 for(var rank=0u;rank<p.y;rank++){
  var best=-3.402823466e38;var token=0xffffffffu;
  for(var t=start+lane;t<end;t+=128u){var skip=false;for(var j=0u;j<rank;j++){skip=skip||(removed[j]==t);}let value=logits[t];if !skip&&(value>best||(value==best&&t>token)){best=value;token=t;}}
  values[lane]=best;tokens[lane]=token;workgroupBarrier();
  for(var step=64u;step>0u;step/=2u){if lane<step{let b=values[lane+step];let bt=tokens[lane+step];if b>values[lane]||(b==values[lane]&&bt>tokens[lane]){values[lane]=b;tokens[lane]=bt;}}workgroupBarrier();}
  if lane==0u{removed[rank]=tokens[0];output[id.x*p.y+rank]=vec2<u32>(tokens[0],bitcast<u32>(values[0]));}workgroupBarrier();
 }
}
