@group(0) @binding(0) var<storage,read> old:array<u32>;
@group(0) @binding(1) var<storage,read_write> next:array<u32>;
@group(0) @binding(2) var<uniform> p:vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;let half=p.x*1024u;
 if i<half {let d=i/p.x;let t=i%p.x;next[d*p.y+t]=old[i];}
 else if i<half*2u {next[p.y*1024u+i-half]=old[i];}
}
