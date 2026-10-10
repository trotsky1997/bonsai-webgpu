@group(0) @binding(0) var<storage,read> old:array<u32>;
@group(0) @binding(1) var<storage,read_write> next:array<u32>;
@group(0) @binding(2) var<uniform> p:vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x+id.y*65535u*256u;let half=p.x*p.w;
 if i<half {let d=i/(p.x*p.z);let offset=i%(p.x*p.z);next[d*p.y*p.z+offset]=old[i];}
 else if i<half*2u {next[p.y*p.w+i-half]=old[i];}
}
