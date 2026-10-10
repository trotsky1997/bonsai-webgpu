// FlashPrefill-inspired: key blocks are pooled on GPU. Microtiles reuse the
// existing 8-token W1A4 graph; this is not the upstream 128-query Triton kernel.
@group(0) @binding(0) var<storage,read> kv:array<u32>;
@group(0) @binding(1) var<storage,read_write> pooled:array<f32>;
@group(0) @binding(2) var<storage,read> state:array<u32>;
@group(0) @binding(3) var<uniform> p:vec4<u32>;
fn nibble(word:u32,i:u32)->f32{return f32(bitcast<i32>((word>>(i*4u))<<28u)>>28);}
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) id:vec3<u32>){
 let count=state[0]+state[1];let begin=id.y*128u;let end=min(begin+128u,count);var sum=0.0;
 for(var t=begin;t<end;t++){let base=((id.x*4u+lane/32u)*p.x+t)*5u;sum+=nibble(kv[base+(lane%32u)/8u],lane%8u)*bitcast<f32>(kv[base+4u]);}
 pooled[(id.x*p.y+id.y)*128u+lane]=sum/f32(max(end-begin,1u));
}
