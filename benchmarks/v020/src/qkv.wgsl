@group(0) @binding(0) var<storage,read> q_w:array<u32>;
@group(0) @binding(1) var<storage,read> q_s:array<f32>;
@group(0) @binding(2) var<storage,read_write> q_out:array<f32>;
@group(0) @binding(3) var<storage,read> k_w:array<u32>;
@group(0) @binding(4) var<storage,read> k_s:array<f32>;
@group(0) @binding(5) var<storage,read_write> k_out:array<f32>;
@group(0) @binding(6) var<storage,read> v_w:array<u32>;
@group(0) @binding(7) var<storage,read> v_s:array<f32>;
@group(0) @binding(8) var<storage,read_write> v_out:array<f32>;
@group(0) @binding(9) var<storage,read> packed:array<u32>;
@group(0) @binding(10) var<uniform> p:vec4<u32>;
fn project_q(row:u32,t:u32) {
 var total=0.0;
 for(var wg=0u;wg<p.y/4u;wg++) {
  let s=q_s[wg*2048u+row];
  for(var j=0u;j<4u;j++) {
   let g=wg*4u+j;let a=((t*p.y)+p.w+g)*8u;
   let r=q_w[g*2048u+row]^packed[a+3u];
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
 q_out[t*2048u+row]=total;
}
fn project_k(row:u32,t:u32) {
 var total=0.0;
 for(var wg=0u;wg<p.y/4u;wg++) {
  let s=k_s[wg*1024u+row];
  for(var j=0u;j<4u;j++) {
   let g=wg*4u+j;let a=((t*p.y)+p.w+g)*8u;
   let r=k_w[g*1024u+row]^packed[a+3u];
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
 k_out[t*1024u+row]=total;
}
fn project_v(row:u32,t:u32) {
 var total=0.0;
 for(var wg=0u;wg<p.y/4u;wg++) {
  let s=v_s[wg*1024u+row];
  for(var j=0u;j<4u;j++) {
   let g=wg*4u+j;let a=((t*p.y)+p.w+g)*8u;
   let r=v_w[g*1024u+row]^packed[a+3u];
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
 v_out[t*1024u+row]=total;
}
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {
 let row=id.x;let t=id.y;
 if row>=0u && row<2048u {project_q(row-0u,t);}
 if row>=2048u && row<3072u {project_k(row-2048u,t);}
 if row>=3072u && row<4096u {project_v(row-3072u,t);}
}
