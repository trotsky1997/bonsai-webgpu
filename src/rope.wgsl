@group(0) @binding(0) var<storage,read_write> q:array<f32>;
@group(0) @binding(1) var<storage,read_write> k:array<f32>;
@group(0) @binding(2) var<storage,read> v:array<f32>;
@group(0) @binding(3) var<storage,read> qw:array<f32>;
@group(0) @binding(4) var<storage,read> kw:array<f32>;
@group(0) @binding(5) var<storage,read_write> kv:array<f32>;
@group(0) @binding(6) var<storage,read> frequencies:array<f32>;
@group(0) @binding(7) var<storage,read> state:array<u32>;
@group(0) @binding(8) var<uniform> p:vec4<u32>;
var<workgroup> values:array<f32,128>;
var<workgroup> sums:array<f32,128>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) id:vec3<u32>) {
 let is_q=id.x<16u;let head=select(id.x-16u,id.x,is_q);let pos=state[0]+id.y;
 var a=0.0;var weight=0.0;
 if is_q {a=q[id.y*2048u+head*128u+lane];weight=qw[lane];}
 else {a=k[id.y*1024u+head*128u+lane];weight=kw[lane];}
 sums[lane]=a*a;workgroupBarrier();
 for(var step=64u;step>0u;step/=2u){if lane<step{sums[lane]+=sums[lane+step];}workgroupBarrier();}
 values[lane]=(a*inverseSqrt(sums[0]/128.0+0.000001))*weight;workgroupBarrier();
 let d=lane%64u;let angle=f32(pos)*frequencies[d];
 let cs=cos(angle)*YARN_ATTN;let sn=sin(angle)*YARN_ATTN;
 var rotated=0.0;
 if lane<64u {rotated=values[d]*cs-values[d+64u]*sn;}
 else {rotated=values[d]*sn+values[d+64u]*cs;}
 if is_q {q[id.y*2048u+head*128u+lane]=rotated;}
 else {kv[(head*128u+lane)*p.x+pos]=rotated;kv[p.x*1024u+pos*1024u+head*128u+lane]=v[id.y*1024u+head*128u+lane];}
}
