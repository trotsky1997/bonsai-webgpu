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
  let s=ws[wg*p.x+row]*0.5;
  let g=wg*4u;
  let a0=((t*p.y)+p.w+g+0u)*8u;
  let r0=weights[(g+0u)*p.x+row]^packed[a0+3u];
  let c0=i32(countOneBits(r0&packed[a0]))+2*i32(countOneBits(r0&packed[a0+1u]))+4*i32(countOneBits(r0&packed[a0+2u]));
  let d0=f32(2*c0-bitcast<i32>(packed[a0+7u]));
  let a1=((t*p.y)+p.w+g+1u)*8u;
  let r1=weights[(g+1u)*p.x+row]^packed[a1+3u];
  let c1=i32(countOneBits(r1&packed[a1]))+2*i32(countOneBits(r1&packed[a1+1u]))+4*i32(countOneBits(r1&packed[a1+2u]));
  let d1=f32(2*c1-bitcast<i32>(packed[a1+7u]));
  let a2=((t*p.y)+p.w+g+2u)*8u;
  let r2=weights[(g+2u)*p.x+row]^packed[a2+3u];
  let c2=i32(countOneBits(r2&packed[a2]))+2*i32(countOneBits(r2&packed[a2+1u]))+4*i32(countOneBits(r2&packed[a2+2u]));
  let d2=f32(2*c2-bitcast<i32>(packed[a2+7u]));
  let a3=((t*p.y)+p.w+g+3u)*8u;
  let r3=weights[(g+3u)*p.x+row]^packed[a3+3u];
  let c3=i32(countOneBits(r3&packed[a3]))+2*i32(countOneBits(r3&packed[a3+1u]))+4*i32(countOneBits(r3&packed[a3+2u]));
  let d3=f32(2*c3-bitcast<i32>(packed[a3+7u]));
  let weighted=vec4<f32>(s)*vec4<f32>(d0,d1,d2,d3);
  let scales=vec4<f32>(bitcast<f32>(packed[a0+4u]),bitcast<f32>(packed[a1+4u]),bitcast<f32>(packed[a2+4u]),bitcast<f32>(packed[a3+4u]));
  let scaled=fma(weighted,scales,vec4<f32>(0.0));
  total=fma(total,1.0,scaled.x);
  total=fma(total,1.0,scaled.y);
  total=fma(total,1.0,scaled.z);
  total=fma(total,1.0,scaled.w);
 }
 output[t*p.x+row]=total;
}
