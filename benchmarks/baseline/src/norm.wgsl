@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read> weight:array<f32>;
@group(0) @binding(2) var<storage,read_write> y:array<f32>;
@group(0) @binding(3) var<uniform> p:vec4<u32>;
var<workgroup> sums:array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) id:vec3<u32>) {
 let offset=id.y*p.x;var total=0.0;
 for(var d=lane;d<p.x;d+=256u){let a=x[offset+d];total+=a*a;}
 sums[lane]=total;workgroupBarrier();
 for(var step=128u;step>0u;step/=2u){if lane<step{sums[lane]+=sums[lane+step];}workgroupBarrier();}
 let inv=inverseSqrt(sums[0]/f32(p.x)+0.000001);
 for(var d=lane;d<p.x;d+=256u){y[offset+d]=(x[offset+d]*inv)*weight[d];}
}
