@group(0) @binding(0) var<storage,read> packed:array<u32>;
@group(0) @binding(1) var<storage,read_write> scales:array<f32>;
@group(0) @binding(2) var<uniform> p:vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>){if id.x<p.x {let value=unpack2x16float(packed[id.x]);scales[id.x*2u]=value.x;scales[id.x*2u+1u]=value.y;}}
