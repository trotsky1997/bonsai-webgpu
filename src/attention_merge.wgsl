@group(0) @binding(0) var<storage,read> parts:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
@group(0) @binding(2) var<uniform> p:vec4<u32>;
var<workgroup> sums:array<f32,128>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) id:vec3<u32>){
 let base=(id.y*16u+id.x)*p.x*130u;var m=-3.402823466e38;
 for(var j=lane;j<p.x;j+=128u){m=max(m,parts[base+j*130u]);}sums[lane]=m;workgroupBarrier();
 for(var step=64u;step>0u;step/=2u){if lane<step{sums[lane]=max(sums[lane],sums[lane+step]);}workgroupBarrier();}
 let maximum=sums[0];var total=0.0;var result=0.0;
 for(var j=0u;j<p.x;j++){let offset=base+j*130u;let scale=exp(parts[offset]-maximum);total+=parts[offset+1u]*scale;result+=parts[offset+2u+lane]*scale;}
 output[id.y*2048u+id.x*128u+lane]=result/max(total,1e-30);
}
