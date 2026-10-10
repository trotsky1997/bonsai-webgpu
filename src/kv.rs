//! INT4 KV codec: signed [-7, 7], 32 values / four words + FP32 scale.
pub fn encode(values:&[f32;32])->[u32;5]{
 let a=values.iter().fold(0.0f32,|m,x|m.max(x.abs()));let mut out=[0;5];out[4]=(a/7.0).to_bits();
 if a>0.0{for(i,x)in values.iter().enumerate(){let q=(x/a*7.0).round().clamp(-7.0,7.0)as i32;out[i/8]|=(q as u32&15)<<(i%8*4);}}out
}
pub fn decode(words:&[u32],d:usize)->f32{let q=((words[d/8]>>(d%8*4))<<28)as i32>>28;q as f32*f32::from_bits(words[4])}
#[cfg(test)]mod tests{use super::*;
 #[test]fn all_codes_and_zero(){for q in -7i32..=7{let v=[q as f32;32];let p=encode(&v);for i in 0..32{assert!((decode(&p,i)-q as f32).abs()<1e-6);}}assert_eq!(encode(&[0.;32]),[0;5]);}
 #[test]fn error_bound(){for seed in 0..3000{let mut v=[0.;32];for(i,x)in v.iter_mut().enumerate(){*x=((seed*97+i*13)%997)as f32/31.-16.;}let p=encode(&v);let s=f32::from_bits(p[4]);for i in 0..32{assert!((decode(&p,i)-v[i]).abs()<=s*0.50001+1e-5);}}}
 #[test]fn full_memory(){assert_eq!(131072u64*28*1280,4697620480);}
}
