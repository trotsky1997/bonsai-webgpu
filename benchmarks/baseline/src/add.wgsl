@group(0) @binding(0) var<storage,read> o:array<f32>;
@group(0) @binding(1) var<storage,read_write> x:array<f32>;
@group(0) @binding(2) var<uniform> p:vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>){if id.x<p.x{x[id.x]=x[id.x]+o[id.x];}}
