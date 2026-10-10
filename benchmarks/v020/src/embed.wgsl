@group(0) @binding(0) var<storage,read> w:array<u32>;
@group(0) @binding(1) var<storage,read> scales:array<f32>;
@group(0) @binding(2) var<storage,read> tokens:array<u32>;
@group(0) @binding(3) var<storage,read_write> x:array<f32>;
@group(0) @binding(4) var<uniform> p:vec4<u32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 let d=id.x;let t=id.y;if d>=p.y{return;}
 let token=tokens[t];let s=scales[(d/128u)*p.x+token];
 let bit=f32((w[(d/32u)*p.x+token]>>(d%32u))&1u);
 x[t*p.y+d]=bit*s-s*0.5;
}
