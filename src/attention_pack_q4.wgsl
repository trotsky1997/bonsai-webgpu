@group(0) @binding(0) var<storage,read> q:array<f32>;
@group(0) @binding(1) var<storage,read> kv:array<u32>;
@group(0) @binding(2) var<storage,read_write> packed:array<u32>;
@group(0) @binding(3) var<storage,read> state:array<u32>;
@group(0) @binding(4) var<uniform> p:vec4<u32>;
var<workgroup> scores:array<f32,CONTEXT_LENGTH>;
var<workgroup> sums:array<f32,128>;
var<workgroup> query:array<f32,128>;
var<workgroup> maxima:array<f32,128>;
var<workgroup> planes:array<vec4<u32>,128>;
var<workgroup> stats:array<vec2<i32>,128>;
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

fn signed_nibble(word:u32,index:u32)->f32{return f32(bitcast<i32>((word>>(index*4u))<<28u)>>28);}
fn key_dot(kh:u32,t:u32)->f32 {
 var dot=0.0;
 for(var g=0u;g<4u;g++) {let base=((kh*4u+g)*p.x+t)*5u;let scale=bitcast<f32>(kv[base+4u]);
  for(var w=0u;w<4u;w++){let bits=kv[base+w];for(var j=0u;j<8u;j++){dot+=query[g*32u+w*8u+j]*(signed_nibble(bits,j)*scale);}}
 }return dot;
}
fn value_at(kh:u32,t:u32,d:u32)->f32 {let base=p.x*160u+t*160u+(kh*4u+d/32u)*5u;return signed_nibble(kv[base+(d%32u)/8u],d%8u)*bitcast<f32>(kv[base+4u]);}

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) id:vec3<u32>) {
 let head=id.x;let kh=head/2u;let count=state[0]+id.y+1u;
 query[lane]=q[id.y*2048u+head*128u+lane];workgroupBarrier();
 var maximum=-3.402823466e38;
 for(var t=lane;t<count;t+=128u){let dot=key_dot(kh,t);let score=dot*0.08838834764831845;scores[t]=score;maximum=max(maximum,score);}
 sums[lane]=maximum;workgroupBarrier();
 for(var step=64u;step>0u;step/=2u){if lane<step{sums[lane]=max(sums[lane],sums[lane+step]);}workgroupBarrier();}
 let maxscore=sums[0];workgroupBarrier();var total=0.0;
 for(var t=lane;t<count;t+=128u){let e=exp(scores[t]-maxscore);scores[t]=e;total+=e;}
 sums[lane]=total;workgroupBarrier();
 for(var step=64u;step>0u;step/=2u){if lane<step{sums[lane]+=sums[lane+step];}workgroupBarrier();}
 let divisor=sums[0];var result=0.0;
 for(var t=0u;t<count;t++){result+=(scores[t]/divisor)*value_at(kh,t,lane);}
 let a=result;
 maxima[lane]=abs(a);workgroupBarrier();
 for(var step=16u;step>0u;step/=2u){if (lane%32u)<step{maxima[lane]=max(maxima[lane],maxima[lane+step]);}workgroupBarrier();}
 let base=(lane/32u)*32u;let amax=maxima[base];var aq=0;
 if amax>0.0 {
  let value=clamp((a/amax)*7.0,-7.0,7.0);let magnitude=abs(value);let integral=floor(magnitude);
  // Avoid floor(abs(x)+0.5), whose addition can round across the half boundary.
  let rounded=integral+select(0.0,1.0,magnitude-integral>=0.5);
  aq=i32(rounded)*select(1,-1,value<0.0);
 }
 let m=u32(abs(aq));let bit=1u<<(lane%32u);
 planes[lane]=vec4<u32>(select(0u,bit,(m&1u)!=0u),select(0u,bit,(m&2u)!=0u),select(0u,bit,(m&4u)!=0u),select(0u,bit,aq<0));
 stats[lane]=vec2<i32>(aq,max(-aq,0));workgroupBarrier();
 for(var step=16u;step>0u;step/=2u){if (lane%32u)<step{planes[lane]=planes[lane]|planes[lane+step];stats[lane]+=stats[lane+step];}workgroupBarrier();}
 if lane%32u==0u {let out=(id.y*64u+head*4u+lane/32u)*8u;packed[out]=planes[lane].x;packed[out+1u]=planes[lane].y;packed[out+2u]=planes[lane].z;packed[out+3u]=planes[lane].w;packed[out+4u]=bitcast<u32>(scale_div7(amax));packed[out+5u]=bitcast<u32>(stats[lane].x);packed[out+6u]=bitcast<u32>(stats[lane].y);packed[out+7u]=bitcast<u32>(stats[lane].x+2*stats[lane].y);}
}
