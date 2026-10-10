// Copy a compact prefix without changing K's group-major or V's token-major layout.
@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read_write> destination:array<u32>;
// source capacity, destination capacity, token count, words per 32-value group
@group(0) @binding(2) var<uniform> p:vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;let width=select(1024u,160u,p.w==5u);let half=p.z*width;
 if i<half {
  let group=i/(p.z*p.w);let offset=i%(p.z*p.w);
  destination[group*p.y*p.w+offset]=source[group*p.x*p.w+offset];
 }else if i<2u*half {
  destination[p.y*width+i-half]=source[p.x*width+i-half];
 }
}
