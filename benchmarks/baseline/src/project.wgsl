@group(0) @binding(0) var<storage,read> weights:array<u32>;
@group(0) @binding(1) var<storage,read> ws:array<f32>;
@group(0) @binding(2) var<storage,read> packed:array<u32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
@group(0) @binding(4) var<uniform> p:vec4<u32>;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 let row=id.x;let t=id.y;if row>=p.x{return;}
 var total=0.0;
 for(var wg=0u;wg<p.y/4u;wg++) {
  let s=ws[wg*p.x+row];let bias=-s*0.5;
  for(var j=0u;j<4u;j++) {
   let g=wg*4u+j;let a=((t*p.y)+p.w+g)*8u;
   let r=weights[g*p.x+row]^packed[a+3u];
   let c=i32(countOneBits(r&packed[a]))+2*i32(countOneBits(r&packed[a+1u]))+4*i32(countOneBits(r&packed[a+2u]));
   let dot=c-bitcast<i32>(packed[a+6u]);
   let weighted=s*f32(dot);let offset=bias*f32(bitcast<i32>(packed[a+5u]));
   // Explicit fma(..., 0) rounds the product before the addition; avoids
   // implicit contraction across the activation scale and total accumulator.
   let scaled=fma(weighted+offset,bitcast<f32>(packed[a+4u]),0.0);
   total=fma(total,1.0,scaled);
  }
 }
 output[t*p.x+row]=total;
}
