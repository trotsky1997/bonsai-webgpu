@group(0) @binding(0) var<storage,read> q:array<f32>;
@group(0) @binding(1) var<storage,read> kv:array<u32>;
@group(0) @binding(2) var<storage,read_write> parts:array<f32>;
@group(0) @binding(3) var<storage,read> state:array<u32>;
@group(0) @binding(4) var<storage,read> selected:array<u32>;
@group(0) @binding(5) var<uniform> p:vec4<u32>;
var<workgroup> scores:array<f32,128>;
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
 let out=((id.y*16u+head)*p.y+id.z)*130u;
 var block=id.z;var valid=true;
 if p.z==1u {let base=head*(p.y+1u);valid=id.z<selected[base];block=selected[base+1u+id.z];}
 let t=block*128u+lane;
 query[lane]=q[id.y*2048u+head*128u+lane];workgroupBarrier();
 var score=-3.402823466e38;if valid&&t<count{score=key_dot(kh,t)*0.08838834764831845;}
 scores[lane]=score;sums[lane]=score;workgroupBarrier();
 for(var step=64u;step>0u;step/=2u){if lane<step{sums[lane]=max(sums[lane],sums[lane+step]);}workgroupBarrier();}
 let m=sums[0];let e=select(0.0,exp(score-m),valid&&t<count);scores[lane]=e;sums[lane]=e;workgroupBarrier();
 for(var step=64u;step>0u;step/=2u){if lane<step{sums[lane]+=sums[lane+step];}workgroupBarrier();}
 if lane==0u{parts[out]=m;parts[out+1u]=sums[0];}
 var result=0.0;if valid{for(var j=0u;j<128u;j++){let jt=block*128u+j;if jt<count{result+=scores[j]*value_at(kh,jt,lane);}}}
 parts[out+2u+lane]=result;
}
