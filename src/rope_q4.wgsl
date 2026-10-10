@group(0) @binding(0) var<storage,read_write> q:array<f32>;
@group(0) @binding(1) var<storage,read_write> k:array<f32>;
@group(0) @binding(2) var<storage,read> v:array<f32>;
@group(0) @binding(3) var<storage,read> qw:array<f32>;
@group(0) @binding(4) var<storage,read> kw:array<f32>;
@group(0) @binding(5) var<storage,read_write> kv:array<u32>;
@group(0) @binding(6) var<storage,read> frequencies:array<f32>;
@group(0) @binding(7) var<storage,read> state:array<u32>;
@group(0) @binding(8) var<uniform> p:vec4<u32>;
var<workgroup> values:array<f32,128>;
var<workgroup> sums:array<f32,128>;
var<workgroup> ka:array<f32,128>;
var<workgroup> va:array<f32,128>;
var<workgroup> km:array<f32,128>;
var<workgroup> vm:array<f32,128>;
var<workgroup> kq:array<i32,128>;
var<workgroup> vq:array<i32,128>;
fn quant(a:f32,m:f32)->i32 {if m==0.0{return 0;}let x=clamp(a/m*7.0,-7.0,7.0);let ax=abs(x);let f=floor(ax);return i32(f+select(0.0,1.0,ax-f>=0.5))*select(1,-1,x<0.0);}
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) id:vec3<u32>) {
 let is_q=id.x<16u;let head=select(id.x-16u,id.x,is_q);let pos=state[0]+id.y;
 var a=0.0;var weight=0.0;
 if is_q {a=q[id.y*2048u+head*128u+lane];weight=qw[lane];}
 else {a=k[id.y*1024u+head*128u+lane];weight=kw[lane];}
 sums[lane]=a*a;workgroupBarrier();
 for(var step=64u;step>0u;step/=2u){if lane<step{sums[lane]+=sums[lane+step];}workgroupBarrier();}
 values[lane]=(a*inverseSqrt(sums[0]/128.0+0.000001))*weight;workgroupBarrier();
 let d=lane%64u;let angle=f32(pos)*frequencies[d];
 let cs=cos(angle)*YARN_ATTN;let sn=sin(angle)*YARN_ATTN;
 var rotated=0.0;
 if lane<64u {rotated=values[d]*cs-values[d+64u]*sn;}
 else {rotated=values[d]*sn+values[d+64u]*cs;}
 if is_q {q[id.y*2048u+head*128u+lane]=rotated;}
 else {
  ka[lane]=rotated;va[lane]=v[id.y*1024u+head*128u+lane];km[lane]=abs(rotated);vm[lane]=abs(va[lane]);workgroupBarrier();
  for(var step=16u;step>0u;step/=2u){if lane%32u<step{km[lane]=max(km[lane],km[lane+step]);vm[lane]=max(vm[lane],vm[lane+step]);}workgroupBarrier();}
  let base=lane/32u*32u;kq[lane]=quant(ka[lane],km[base]);vq[lane]=quant(va[lane],vm[base]);workgroupBarrier();
  let group=head*4u+lane/32u;let ko=(group*p.x+pos)*5u;let vo=p.x*160u+pos*160u+group*5u;
  if lane%8u==0u {var kb=0u;var vb=0u;for(var j=0u;j<8u;j++){kb|=(u32(kq[lane+j])&15u)<<(j*4u);vb|=(u32(vq[lane+j])&15u)<<(j*4u);}kv[ko+lane%32u/8u]=kb;kv[vo+lane%32u/8u]=vb;}
  if lane%32u==0u {kv[ko+4u]=bitcast<u32>(km[base]/7.0);kv[vo+4u]=bitcast<u32>(vm[base]/7.0);}
 }
}
