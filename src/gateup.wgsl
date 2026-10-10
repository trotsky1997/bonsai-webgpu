@group(0) @binding(0) var<storage,read> gate_w:array<u32>;
@group(0) @binding(1) var<storage,read> gate_s:array<f32>;
@group(0) @binding(2) var<storage,read_write> gate_out:array<f32>;
@group(0) @binding(3) var<storage,read> up_w:array<u32>;
@group(0) @binding(4) var<storage,read> up_s:array<f32>;
@group(0) @binding(5) var<storage,read_write> up_out:array<f32>;
@group(0) @binding(6) var<storage,read> packed:array<u32>;
@group(0) @binding(7) var<uniform> p:vec4<u32>;
fn project_gate(row:u32,t:u32) {
 var total=0.0;
 for(var wg=0u;wg<p.y/4u;wg++) {
  let s=gate_s[wg*6144u+row];
  for(var j=0u;j<4u;j++) {
   let g=wg*4u+j;let a=((t*p.y)+p.w+g)*8u;
   let r=gate_w[g*6144u+row]^packed[a+3u];
   let c=i32(countOneBits(r&packed[a]))+2*i32(countOneBits(r&packed[a+1u]))+4*i32(countOneBits(r&packed[a+2u]));
   let dot=2*c-bitcast<i32>(packed[a+7u]);
   // FP16 scale times this small integer is exactly representable in FP32.
   // Thus centered affine multiplication preserves every original result.
   let weighted=(s*0.5)*f32(dot);
   // Explicit fma(..., 0) rounds the product before the addition; avoids
   // implicit contraction across the activation scale and total accumulator.
   let scaled=fma(weighted,bitcast<f32>(packed[a+4u]),0.0);
   total=fma(total,1.0,scaled);
  }
 }
 gate_out[t*6144u+row]=total;
}
fn project_up(row:u32,t:u32) {
 var total=0.0;
 for(var wg=0u;wg<p.y/4u;wg++) {
  let s=up_s[wg*6144u+row];
  for(var j=0u;j<4u;j++) {
   let g=wg*4u+j;let a=((t*p.y)+p.w+g)*8u;
   let r=up_w[g*6144u+row]^packed[a+3u];
   let c=i32(countOneBits(r&packed[a]))+2*i32(countOneBits(r&packed[a+1u]))+4*i32(countOneBits(r&packed[a+2u]));
   let dot=2*c-bitcast<i32>(packed[a+7u]);
   // FP16 scale times this small integer is exactly representable in FP32.
   // Thus centered affine multiplication preserves every original result.
   let weighted=(s*0.5)*f32(dot);
   // Explicit fma(..., 0) rounds the product before the addition; avoids
   // implicit contraction across the activation scale and total accumulator.
   let scaled=fma(weighted,bitcast<f32>(packed[a+4u]),0.0);
   total=fma(total,1.0,scaled);
  }
 }
 up_out[t*6144u+row]=total;
}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 let row=id.x;let t=id.y;
 if row>=0u && row<6144u {project_gate(row-0u,t);}
 if row>=6144u && row<12288u {project_up(row-6144u,t);}
}
