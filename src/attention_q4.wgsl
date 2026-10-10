@group(0) @binding(0) var<storage,read> q:array<f32>;
@group(0) @binding(1) var<storage,read> kv:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read> state:array<u32>;
@group(0) @binding(4) var<uniform> p:vec4<u32>;
var<workgroup> scores:array<f32,CONTEXT_LENGTH>;
var<workgroup> sums:array<f32,128>;
var<workgroup> query:array<f32,128>;
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
 output[id.y*2048u+head*128u+lane]=result;
}
