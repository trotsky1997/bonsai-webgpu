@group(0) @binding(0) var<storage,read> x:array<f32>;
@group(0) @binding(1) var<storage,read_write> packed:array<u32>;
@group(0) @binding(2) var<uniform> p:vec4<u32>;
var<workgroup> maxima:array<f32,32>;
var<workgroup> planes:array<vec4<u32>,32>;
var<workgroup> stats:array<vec2<i32>,32>;
// Correctly rounded positive normal FP32 / 7. WebGPU permits division to be
// lowered to reciprocal multiplication; that changes A4 scales by one ULP.
fn scale_div7(x:f32)->f32 {
 if x==0.0{return 0.0;}
 let bits=bitcast<u32>(x);let exponent=(bits>>23u)&255u;
 if exponent<=3u{let significand=select(bits&0x7fffffu,(bits&0x7fffffu)|0x800000u,exponent>0u);let numerator=significand<<select(0u,exponent-1u,exponent>0u);let quotient=numerator/7u;return bitcast<f32>(quotient+select(0u,1u,numerator%7u>=4u));}
 let mantissa=(bits&0x7fffffu)|0x800000u;
 let shift=select(3u,2u,mantissa>=14680064u);
 let numerator=mantissa<<shift;let quotient=numerator/7u;let remainder=numerator%7u;
 let rounded=quotient+select(0u,1u,remainder>=4u);
 return bitcast<f32>(((exponent-shift)<<23u)|(rounded-0x800000u));
}
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) id:vec3<u32>) {
 let g=id.y*p.x+id.x;let a=x[g*32u+lane];
 maxima[lane]=abs(a);workgroupBarrier();
 for(var step=16u;step>0u;step/=2u){if lane<step{maxima[lane]=max(maxima[lane],maxima[lane+step]);}workgroupBarrier();}
 let maximum=maxima[0];var q=0;
 if maximum>0.0 {
  let value=clamp((a/maximum)*7.0,-7.0,7.0);let magnitude=abs(value);let integral=floor(magnitude);
  // Avoid floor(abs(x)+0.5), whose addition can round across the half boundary.
  let rounded=integral+select(0.0,1.0,magnitude-integral>=0.5);
  q=i32(rounded)*select(1,-1,value<0.0);
 }
 let m=u32(abs(q));let bit=1u<<lane;
 planes[lane]=vec4<u32>(select(0u,bit,(m&1u)!=0u),select(0u,bit,(m&2u)!=0u),select(0u,bit,(m&4u)!=0u),select(0u,bit,q<0));
 stats[lane]=vec2<i32>(q,max(-q,0));workgroupBarrier();
 for(var step=16u;step>0u;step/=2u){if lane<step{planes[lane]=planes[lane]|planes[lane+step];stats[lane]+=stats[lane+step];}workgroupBarrier();}
 if lane==0u {let out=g*8u;packed[out]=planes[0].x;packed[out+1u]=planes[0].y;packed[out+2u]=planes[0].z;packed[out+3u]=planes[0].w;packed[out+4u]=bitcast<u32>(scale_div7(maximum));packed[out+5u]=bitcast<u32>(stats[0].x);packed[out+6u]=bitcast<u32>(stats[0].y);packed[out+7u]=bitcast<u32>(stats[0].x+2*stats[0].y);}
}
