@group(0) @binding(0) var<storage,read> q:array<f32>;
@group(0) @binding(1) var<storage,read> kv:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
@group(0) @binding(3) var<storage,read> state:array<u32>;
@group(0) @binding(4) var<uniform> p:vec4<u32>;
var<workgroup> scores:array<f32,CONTEXT_LENGTH>;
var<workgroup> sums:array<f32,128>;
var<workgroup> query:array<f32,128>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) id:vec3<u32>) {
 let head=id.x;let kh=head/2u;let count=state[0]+id.y+1u;
 query[lane]=q[id.y*2048u+head*128u+lane];workgroupBarrier();
 var maximum=-3.402823466e38;
 for(var t=lane;t<count;t+=128u){var dot=0.0;for(var d=0u;d<128u;d++){dot+=query[d]*kv[t*1024u+kh*128u+d];}let score=dot*0.08838834764831845;scores[t]=score;maximum=max(maximum,score);}
 sums[lane]=maximum;workgroupBarrier();
 for(var step=64u;step>0u;step/=2u){if lane<step{sums[lane]=max(sums[lane],sums[lane+step]);}workgroupBarrier();}
 let maxscore=sums[0];workgroupBarrier();var total=0.0;
 for(var t=lane;t<count;t+=128u){let e=exp(scores[t]-maxscore);scores[t]=e;total+=e;}
 sums[lane]=total;workgroupBarrier();
 for(var step=64u;step>0u;step/=2u){if lane<step{sums[lane]+=sums[lane+step];}workgroupBarrier();}
 let divisor=sums[0];var result=0.0;
 for(var t=0u;t<count;t++){result+=(scores[t]/divisor)*kv[p.x*1024u+t*1024u+kh*128u+lane];}
 output[id.y*2048u+head*128u+lane]=result;
}
