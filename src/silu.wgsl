@group(0) @binding(0) var<storage,read> gate:array<f32>;
@group(0) @binding(1) var<storage,read> up:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<uniform> p:vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
 if id.x<p.x{
  let g=gate[id.x];
  // Match MLX's stable sigmoid, including the positive-branch subtraction.
  let y=1.0/(1.0+exp(abs(g)));let sigmoid=select(1.0-y,y,g<0.0);
  let silu=fma(g,sigmoid,0.0);output[id.x]=fma(silu,up[id.x],0.0);
 }
}
