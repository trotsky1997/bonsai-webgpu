@group(0) @binding(0) var<storage,read> q:array<f32>;
@group(0) @binding(1) var<storage,read> pooled:array<f32>;
@group(0) @binding(2) var<storage,read_write> selected:array<u32>;
@group(0) @binding(3) var<storage,read> state:array<u32>;
@group(0) @binding(4) var<uniform> p:vec4<u32>;
var<workgroup> scores:array<f32,1024>;
var<workgroup> maxima:array<f32,128>;
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) id:vec3<u32>){
 let head=id.x;let kh=head/2u;var peak=-3.402823466e38;
 for(var block=lane;block<p.y;block+=128u){
  var m=-3.402823466e38;var total=0.0;
  for(var row=0u;row<state[1];row++){
   var dot=0.0;for(var d=0u;d<128u;d++){dot+=q[row*2048u+head*128u+d]*pooled[(kh*p.y+block)*128u+d];}
   let score=dot*0.08838834764831845;let next=max(m,score);total=total*exp(m-next)+exp(score-next);m=next;
  }
  let energy=m+log(max(total,1e-30));scores[block]=energy;peak=max(peak,energy);
 }
 maxima[lane]=peak;workgroupBarrier();
 for(var step=64u;step>0u;step/=2u){if lane<step{maxima[lane]=max(maxima[lane],maxima[lane+step]);}workgroupBarrier();}
 if lane==0u {
  let threshold=maxima[0]+log(bitcast<f32>(p.w));let base=head*(p.y+1u);var n=0u;
  for(var block=0u;block<p.y;block++){if block<2u||block+4u>=p.y||scores[block]>=threshold{selected[base+1u+n]=block;n++;}}
  selected[base]=n;
 }
}
