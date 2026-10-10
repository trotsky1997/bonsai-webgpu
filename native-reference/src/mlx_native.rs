//! Rust model executor with native MLX arrays and our packed W1A4 Metal kernels.
//! Arrays stay on the inference thread; only final logits return to the host.
use super::*;
use std::{
    ffi::{c_char, c_void, CStr, CString},
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
};

type Handle = *mut c_void;
extern "C" {
    fn bn_active_memory() -> usize;
    fn bn_peak_memory() -> usize;
    fn bn_cache_memory() -> usize;
    fn bn_reset_peak_memory();
    fn bn_error() -> *const c_char;
    fn bn_initialize(metallib_path: *const c_char) -> i32;
    fn bn_free(a: Handle);
    fn bn_f32(data: *const f32, dims: *const i32, rank: i32) -> Handle;
    fn bn_u32(data: *const u32, dims: *const i32, rank: i32) -> Handle;
    fn bn_eval(a: Handle) -> i32;
    fn bn_read(a: Handle, output: *mut f32, count: usize) -> i32;
    fn bn_read_bits(a: Handle, output: *mut u32, count: usize) -> i32;
    fn bn_norm(a: Handle, weight: Handle, eps: f32, head_dim: i32) -> Handle;
    fn bn_add(a: Handle, b: Handle) -> Handle;
    fn bn_silu_mul(gate: Handle, up: Handle) -> Handle;
    fn bn_pack(x: Handle, groups: i32, outputs: *mut Handle) -> i32;
    fn bn_project(
        w: Handle,
        ws: Handle,
        wb: Handle,
        planes: Handle,
        scales: Handle,
        stats: Handle,
        n: i32,
        g: i32,
    ) -> Handle;
    fn bn_embed(w: Handle, ws: Handle, wb: Handle, token: Handle, n: i32, k: i32) -> Handle;
    fn bn_rope(
        x: Handle,
        frequencies: Handle,
        scale: Handle,
        position: Handle,
        nh: i32,
        hd: i32,
    ) -> Handle;
    fn bn_cache_new(nk: i32, capacity: i32, hd: i32) -> Handle;
    fn bn_cache_grow(old: Handle, nk: i32, extra: i32, hd: i32) -> Handle;
    fn bn_cache_put(old: Handle, value: Handle, nk: i32, pos: i32, hd: i32) -> Handle;
    fn bn_attention(q: Handle, k: Handle, v: Handle, nq: i32, nk: i32, seq: i32, hd: i32)
        -> Handle;
}
fn error() -> anyhow::Error {
    unsafe { anyhow::anyhow!("MLX: {}", CStr::from_ptr(bn_error()).to_string_lossy()) }
}
fn status(code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(error())
    }
}

struct Array {
    ptr: NonNull<c_void>,
    _thread_bound: PhantomData<Rc<()>>,
}
impl Drop for Array {
    fn drop(&mut self) {
        unsafe {
            bn_free(self.raw());
        }
    }
}
impl Array {
    fn take(ptr: Handle) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or_else(error)?,
            _thread_bound: PhantomData,
        })
    }
    fn raw(&self) -> Handle {
        self.ptr.as_ptr()
    }
    fn f32(data: &[f32], dims: &[i32]) -> Result<Self> {
        if dims.iter().map(|x| *x as usize).product::<usize>() != data.len() {
            bail!("MLX float array size mismatch");
        }
        Self::take(unsafe { bn_f32(data.as_ptr(), dims.as_ptr(), dims.len() as i32) })
    }
    fn u32(data: &[u32], dims: &[i32]) -> Result<Self> {
        if dims.iter().map(|x| *x as usize).product::<usize>() != data.len() {
            bail!("MLX integer array size mismatch");
        }
        Self::take(unsafe { bn_u32(data.as_ptr(), dims.as_ptr(), dims.len() as i32) })
    }
    fn eval(&self) -> Result<()> {
        status(unsafe { bn_eval(self.raw()) })
    }
    fn norm(&self, weight: &Self, eps: f32, hd: usize) -> Result<Self> {
        Self::take(unsafe { bn_norm(self.raw(), weight.raw(), eps, hd as i32) })
    }
    fn add(&self, b: &Self) -> Result<Self> {
        Self::take(unsafe { bn_add(self.raw(), b.raw()) })
    }
    fn pack(&self, groups: usize) -> Result<Packed> {
        let mut handles = [std::ptr::null_mut(); 3];
        status(unsafe { bn_pack(self.raw(), groups as i32, handles.as_mut_ptr()) })?;
        Ok(Packed {
            planes: Self::take(handles[0])?,
            scales: Self::take(handles[1])?,
            stats: Self::take(handles[2])?,
        })
    }
}
struct Packed {
    planes: Array,
    scales: Array,
    stats: Array,
}
fn trace_array(enabled: bool, name: &str, a: &Array, count: usize) -> Result<()> {
    if !enabled {
        return Ok(());
    }
    let mut values = vec![0f32; count];
    status(unsafe { bn_read(a.raw(), values.as_mut_ptr(), count) })?;
    let path = std::env::var("BONSAI_TRACE").context("trace path")?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    serde_json::to_writer(&mut file, &serde_json::json!({"name":name,"values":values}))?;
    file.write_all(b"\n")?;
    Ok(())
}
fn trace_packed(enabled: bool, name: &str, p: &Packed, groups: usize) -> Result<()> {
    if !enabled {
        return Ok(());
    }
    let mut planes = vec![0u32; groups * 4];
    let mut scales = vec![0f32; groups];
    let mut stats = vec![0u32; groups * 2];
    status(unsafe { bn_read_bits(p.planes.raw(), planes.as_mut_ptr(), planes.len()) })?;
    status(unsafe { bn_read(p.scales.raw(), scales.as_mut_ptr(), scales.len()) })?;
    status(unsafe { bn_read_bits(p.stats.raw(), stats.as_mut_ptr(), stats.len()) })?;
    let values: Vec<u32> = (0..groups)
        .flat_map(|i| {
            [
                planes[i * 4],
                planes[i * 4 + 1],
                planes[i * 4 + 2],
                planes[i * 4 + 3],
                scales[i].to_bits(),
                stats[i * 2],
                stats[i * 2 + 1],
                0,
            ]
        })
        .collect();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(std::env::var("BONSAI_TRACE")?)?;
    serde_json::to_writer(&mut file, &serde_json::json!({"name":name,"values":values}))?;
    file.write_all(b"\n")?;
    Ok(())
}
struct Matrix {
    w: Array,
    scales: Array,
    biases: Array,
    input: usize,
    out: usize,
}
impl Matrix {
    fn load(sf: &SafeFile, name: &str) -> Result<Self> {
        let linear = sf.linear(name)?;
        if linear.input % 128 != 0 {
            bail!("{name}: native W1A4 requires weight groups of 128");
        }
        let n = linear.out;
        let g = linear.input / 32;
        // Transpose once: adjacent GPU threads read adjacent output channels.
        let mut words = vec![0u32; n * g];
        for row in 0..n {
            for group in 0..g {
                let offset = (row * g + group) * 4;
                words[group * n + row] =
                    u32::from_le_bytes(linear.packed[offset..offset + 4].try_into().unwrap());
            }
        }
        let transpose = |data: &[f32]| {
            let mut result = vec![0.0; data.len()];
            for row in 0..n {
                for group in 0..g / 4 {
                    result[group * n + row] = data[row * (g / 4) + group];
                }
            }
            result
        };
        Ok(Self {
            w: Array::u32(&words, &[g as i32, n as i32])?,
            scales: Array::f32(&transpose(&linear.scales), &[(g / 4) as i32, n as i32])?,
            biases: Array::f32(&transpose(&linear.biases), &[(g / 4) as i32, n as i32])?,
            input: linear.input,
            out: n,
        })
    }
    fn project(&self, p: &Packed) -> Result<Array> {
        Array::take(unsafe {
            bn_project(
                self.w.raw(),
                self.scales.raw(),
                self.biases.raw(),
                p.planes.raw(),
                p.scales.raw(),
                p.stats.raw(),
                self.out as i32,
                (self.input / 32) as i32,
            )
        })
    }
    fn embed(&self, token: &Array) -> Result<Array> {
        Array::take(unsafe {
            bn_embed(
                self.w.raw(),
                self.scales.raw(),
                self.biases.raw(),
                token.raw(),
                self.out as i32,
                self.input as i32,
            )
        })
    }
}
struct GpuLayer {
    input_norm: Array,
    post_norm: Array,
    q_norm: Array,
    k_norm: Array,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
    keys: Option<Array>,
    values: Option<Array>,
    capacity: usize,
}
pub(super) struct MlxModel {
    cfg: Config,
    tok: Tokenizer,
    embedding: Matrix,
    norm: Array,
    layers: Vec<GpuLayer>,
    frequencies: Array,
    attention_scale: Array,
    pos: usize,
}
impl MlxModel {
    pub(super) fn load(dir: &Path) -> Result<Self> {
        let library = std::env::var_os("BONSAI_MLX_METALLIB")
            .map(PathBuf::from)
            .unwrap_or(
                std::env::current_exe()?
                    .parent()
                    .context("binary directory missing")?
                    .join("../lib/mlx.metallib"),
            );
        let library = std::fs::canonicalize(&library)
            .with_context(|| format!("missing MLX Metal library: {}", library.display()))?;
        let path = CString::new(library.to_string_lossy().as_bytes())?;
        status(unsafe { bn_initialize(path.as_ptr()) })?;
        let cfg: Config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        if cfg.hidden_size != cfg.num_attention_heads * cfg.head_dim
            || cfg.head_dim % 2 != 0
            || cfg.num_key_value_heads == 0
            || cfg.num_attention_heads % cfg.num_key_value_heads != 0
        {
            bail!("unsupported attention dimensions");
        }
        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let sf = SafeFile::open(&dir.join("model.safetensors"))?;
        let embedding = Matrix::load(&sf, "model.embed_tokens")?;
        if embedding.out != cfg.vocab_size || embedding.input != cfg.hidden_size {
            bail!("embedding dimensions do not match config");
        }
        let norm = |name: &str, size: usize| -> Result<Array> {
            Array::f32(&sf.halfs(name)?, &[size as i32])
        };
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let l = GpuLayer {
                input_norm: norm(&format!("{p}.input_layernorm.weight"), cfg.hidden_size)?,
                post_norm: norm(
                    &format!("{p}.post_attention_layernorm.weight"),
                    cfg.hidden_size,
                )?,
                q_norm: norm(&format!("{p}.self_attn.q_norm.weight"), cfg.head_dim)?,
                k_norm: norm(&format!("{p}.self_attn.k_norm.weight"), cfg.head_dim)?,
                q: Matrix::load(&sf, &format!("{p}.self_attn.q_proj"))?,
                k: Matrix::load(&sf, &format!("{p}.self_attn.k_proj"))?,
                v: Matrix::load(&sf, &format!("{p}.self_attn.v_proj"))?,
                o: Matrix::load(&sf, &format!("{p}.self_attn.o_proj"))?,
                gate: Matrix::load(&sf, &format!("{p}.mlp.gate_proj"))?,
                up: Matrix::load(&sf, &format!("{p}.mlp.up_proj"))?,
                down: Matrix::load(&sf, &format!("{p}.mlp.down_proj"))?,
                keys: None,
                values: None,
                capacity: 0,
            };
            for (m, input, out) in [
                (&l.q, cfg.hidden_size, cfg.hidden_size),
                (
                    &l.k,
                    cfg.hidden_size,
                    cfg.num_key_value_heads * cfg.head_dim,
                ),
                (
                    &l.v,
                    cfg.hidden_size,
                    cfg.num_key_value_heads * cfg.head_dim,
                ),
                (&l.o, cfg.hidden_size, cfg.hidden_size),
                (&l.gate, cfg.hidden_size, cfg.intermediate_size),
                (&l.up, cfg.hidden_size, cfg.intermediate_size),
                (&l.down, cfg.intermediate_size, cfg.hidden_size),
            ] {
                if m.input != input || m.out != out {
                    bail!("layer {i}: projection dimensions do not match config");
                }
            }
            layers.push(l);
        }
        let half = cfg.head_dim / 2;
        let low_high = cfg.rope_scaling.as_ref().map(|y| {
            let corr = |rotations: f32| {
                (cfg.head_dim as f32
                    * (y.original_max_position_embeddings as f32
                        / (rotations * 2.0 * std::f32::consts::PI))
                        .ln())
                    / (2.0 * cfg.rope_theta.ln())
            };
            (
                corr(32.0).floor().clamp(0.0, (half - 1) as f32),
                corr(1.0).ceil().clamp(0.0, (half - 1) as f32),
            )
        });
        let frequencies: Vec<f32> = (0..half)
            .map(|i| {
                let inv = cfg.rope_theta.powf(-((2 * i) as f32) / cfg.head_dim as f32);
                if let (Some(y), Some((low, high))) = (cfg.rope_scaling.as_ref(), low_high) {
                    let ramp = ((i as f32 - low) / if low == high { 0.001 } else { high - low })
                        .clamp(0.0, 1.0);
                    inv / y.factor * ramp + inv * (1.0 - ramp)
                } else {
                    inv
                }
            })
            .collect();
        let attention_scale = cfg
            .rope_scaling
            .as_ref()
            .map(|y| 1.0 + 0.1 * y.factor.ln())
            .unwrap_or(1.0);
        Ok(Self {
            norm: norm("model.norm.weight", cfg.hidden_size)?,
            tok,
            embedding,
            layers,
            frequencies: Array::f32(&frequencies, &[half as i32])?,
            attention_scale: Array::f32(&[attention_scale], &[1])?,
            cfg,
            pos: 0,
        })
    }
}
impl chat_worker::ChatModel for MlxModel {
    fn config(&self) -> &Config {
        &self.cfg
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }
    fn truncate_cache(&mut self, prefix: usize) -> Result<()> {
        if prefix > self.pos {
            bail!("invalid KV cache prefix");
        }
        self.pos = prefix;
        Ok(())
    }
    fn reset_cache(&mut self) {
        self.pos = 0;
        for l in &mut self.layers {
            l.keys = None;
            l.values = None;
            l.capacity = 0;
        }
        unsafe {
            bn_reset_peak_memory();
        }
    }
    fn memory_info(&self) -> serde_json::Value {
        unsafe {
            serde_json::json!({"mlx_active_bytes":bn_active_memory(),"mlx_peak_bytes":bn_peak_memory(),"mlx_cache_bytes":bn_cache_memory(),"kv_capacity_tokens":self.layers.first().map(|l|l.capacity).unwrap_or(0),"used_tokens":self.pos})
        }
    }
    fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let c = &self.cfg;
        if self.pos >= c.max_position_embeddings || token as usize >= c.vocab_size {
            bail!("token or position out of range");
        }
        let token = Array::u32(&[token], &[1])?;
        let position = Array::u32(&[self.pos as u32], &[1])?;
        let mut x = self.embedding.embed(&token)?;
        let tracing = self.pos == 0 && std::env::var_os("BONSAI_TRACE").is_some();
        trace_array(tracing, "embed", &x, c.hidden_size)?;
        for (layer_index, l) in self.layers.iter_mut().enumerate() {
            let tr = tracing && layer_index == 0;
            let z = x.norm(&l.input_norm, c.rms_norm_eps, 0)?;
            trace_array(tr, "norm1", &z, c.hidden_size)?;
            let z = z.pack(c.hidden_size / 32)?;
            trace_packed(tr, "pack1", &z, c.hidden_size / 32)?;
            let q = l.q.project(&z)?;
            trace_array(tr, "q", &q, c.hidden_size)?;
            let q = q.norm(&l.q_norm, c.rms_norm_eps, c.head_dim)?;
            let k = l.k.project(&z)?;
            trace_array(tr, "k", &k, c.num_key_value_heads * c.head_dim)?;
            let k = k.norm(&l.k_norm, c.rms_norm_eps, c.head_dim)?;
            let v = l.v.project(&z)?;
            trace_array(tr, "v", &v, c.num_key_value_heads * c.head_dim)?;
            let rotate = |a: &Array, heads: usize| {
                Array::take(unsafe {
                    bn_rope(
                        a.raw(),
                        self.frequencies.raw(),
                        self.attention_scale.raw(),
                        position.raw(),
                        heads as i32,
                        c.head_dim as i32,
                    )
                })
            };
            let q = rotate(&q, c.num_attention_heads)?;
            trace_array(tr, "rope_q", &q, c.hidden_size)?;
            let k = rotate(&k, c.num_key_value_heads)?;
            let nk = c.num_key_value_heads as i32;
            let hd = c.head_dim as i32;
            if self.pos >= l.capacity {
                let next = (l.capacity + 256).min(c.max_position_embeddings);
                for cache in [&mut l.keys, &mut l.values] {
                    *cache = Some(Array::take(unsafe {
                        match cache.as_ref() {
                            Some(old) => {
                                bn_cache_grow(old.raw(), nk, (next - l.capacity) as i32, hd)
                            }
                            None => bn_cache_new(nk, next as i32, hd),
                        }
                    })?);
                }
                l.capacity = next;
            }
            l.keys = Some(Array::take(unsafe {
                bn_cache_put(
                    l.keys.as_ref().unwrap().raw(),
                    k.raw(),
                    nk,
                    self.pos as i32,
                    hd,
                )
            })?);
            l.values = Some(Array::take(unsafe {
                bn_cache_put(
                    l.values.as_ref().unwrap().raw(),
                    v.raw(),
                    nk,
                    self.pos as i32,
                    hd,
                )
            })?);
            let att = Array::take(unsafe {
                bn_attention(
                    q.raw(),
                    l.keys.as_ref().unwrap().raw(),
                    l.values.as_ref().unwrap().raw(),
                    c.num_attention_heads as i32,
                    nk,
                    (self.pos + 1) as i32,
                    hd,
                )
            })?;
            trace_array(tr, "attention", &att, c.hidden_size)?;
            let pa = att.pack(c.hidden_size / 32)?;
            trace_packed(tr, "pack_att", &pa, c.hidden_size / 32)?;
            let o = l.o.project(&pa)?;
            trace_array(tr, "o", &o, c.hidden_size)?;
            x = x.add(&o)?;
            trace_array(tr, "residual1", &x, c.hidden_size)?;
            let z = x.norm(&l.post_norm, c.rms_norm_eps, 0)?;
            trace_array(tr, "norm2", &z, c.hidden_size)?;
            let z = z.pack(c.hidden_size / 32)?;
            trace_packed(tr, "pack2", &z, c.hidden_size / 32)?;
            let gate = l.gate.project(&z)?;
            trace_array(tr, "gate", &gate, c.intermediate_size)?;
            let up = l.up.project(&z)?;
            trace_array(tr, "up", &up, c.intermediate_size)?;
            let activation = Array::take(unsafe { bn_silu_mul(gate.raw(), up.raw()) })?;
            trace_array(tr, "silu", &activation, c.intermediate_size)?;
            let pa = activation.pack(c.intermediate_size / 32)?;
            trace_packed(tr, "pack_down", &pa, c.intermediate_size / 32)?;
            let down = l.down.project(&pa)?;
            trace_array(tr, "down", &down, c.hidden_size)?;
            x = x.add(&down)?;
            trace_array(tr, "residual2", &x, c.hidden_size)?;
        }
        let logits = self.embedding.project(
            &x.norm(&self.norm, c.rms_norm_eps, 0)?
                .pack(c.hidden_size / 32)?,
        )?;
        let mut result = vec![0.0; c.vocab_size];
        status(unsafe { bn_read(logits.raw(), result.as_mut_ptr(), result.len()) })?;
        // Materialize cache roots so the next token cannot retain the old graph.
        for l in &self.layers {
            l.keys.as_ref().unwrap().eval()?;
            l.values.as_ref().unwrap().eval()?;
        }
        if result.iter().any(|v| !v.is_finite()) {
            bail!("non-finite MLX logits");
        }
        self.pos += 1;
        Ok(result)
    }
}
