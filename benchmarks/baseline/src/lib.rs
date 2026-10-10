//! Dedicated Bonsai W1A4 browser executor. No native service or generic ML runtime.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;

fn err(s: impl ToString) -> JsValue {
    JsValue::from_str(&s.to_string())
}
type Result<T> = std::result::Result<T, JsValue>;

#[derive(Deserialize)]
struct Blob {
    matrices: Vec<MatInfo>,
    vectors: Vec<VecInfo>,
}
#[derive(Deserialize)]
struct MatInfo {
    name: String,
    n: u32,
    k: u32,
    w: [usize; 2],
    s: [usize; 2],
}
#[derive(Deserialize)]
struct VecInfo {
    name: String,
    data: [usize; 2],
}
struct Matrix {
    weights: wgpu::Buffer,
    scales: wgpu::Buffer,
    n: u32,
    k: u32,
}
struct Kernel {
    pipeline: wgpu::ComputePipeline,
}
struct Dispatch {
    pipeline: String,
    group: wgpu::BindGroup,
    dims: [u32; 3],
}
struct Layer {
    matrices: HashMap<String, Matrix>,
    vectors: HashMap<String, wgpu::Buffer>,
    kv: wgpu::Buffer,
}
#[derive(Serialize)]
struct Memory {
    gpu_weights_bytes: u64,
    gpu_kv_bytes: u64,
    gpu_scratch_bytes: u64,
    wasm_linear_bytes: u64,
    context: u32,
    batch: u32,
    kv_capacity_tokens: u32,
    used_tokens: u32,
}

#[wasm_bindgen]
pub struct Engine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    tokenizer: Tokenizer,
    matrices: HashMap<String, Matrix>,
    vectors: HashMap<String, wgpu::Buffer>,
    layers: Vec<Layer>,
    kernels: HashMap<String, Kernel>,
    buffers: HashMap<String, wgpu::Buffer>,
    graphs: HashMap<u32, Vec<Dispatch>>,
    capacity: u32,
    kv_capacity: u32,
    batch: u32,
    pos: u32,
    weights_bytes: u64,
    kv_bytes: u64,
    scratch_bytes: u64,
    adapter_info: String,
}

impl Engine {
    async fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Result<Vec<u32>> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &self.buffers["readback"], 0, size);
        self.queue.submit(Some(encoder.finish()));
        let slice = self.buffers["readback"].slice(..size);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        rx.await.map_err(err)?.map_err(err)?;
        let data = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
        self.buffers["readback"].unmap();
        Ok(data)
    }
    fn buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(4),
            usage,
            mapped_at_creation: false,
        })
    }
    fn data(&self, label: &str, bytes: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
    }
    fn uniform(&self, data: &[u32]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shape"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }
    fn dispatch(
        &self,
        name: &str,
        bufs: &[&wgpu::Buffer],
        dims: [u32; 3],
        shape: [u32; 4],
    ) -> Dispatch {
        let p = &self.kernels[name].pipeline;
        let u = self.uniform(&shape);
        let mut entries: Vec<_> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: bufs.len() as u32,
            resource: u.as_entire_binding(),
        });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(name),
            layout: &p.get_bind_group_layout(0),
            entries: &entries,
        });
        Dispatch {
            pipeline: name.into(),
            group,
            dims,
        }
    }
    fn project(&self, m: &Matrix, out: &wgpu::Buffer, b: u32, tail: bool) -> Dispatch {
        let count = if tail { 1 } else { b };
        self.dispatch(
            "project",
            &[&m.weights, &m.scales, &self.buffers["packed"], out],
            [(m.n + 127) / 128, count, 1],
            [
                m.n,
                m.k / 32,
                count,
                if tail { (b - 1) * (m.k / 32) } else { 0 },
            ],
        )
    }
    fn graph(&self, b: u32) -> Vec<Dispatch> {
        let s = &self.buffers;
        let mut v = Vec::new();
        v.push(self.dispatch(
            "embed",
            &[
                &self.matrices["embed"].weights,
                &self.matrices["embed"].scales,
                &s["tokens"],
                &s["x"],
            ],
            [16, b, 1],
            [151669, 2048, b, 0],
        ));
        let norm = |weight: &wgpu::Buffer| {
            self.dispatch(
                "norm",
                &[&s["x"], weight, &s["z"]],
                [1, b, 1],
                [2048, b, 0, 0],
            )
        };
        let pack = |input: &wgpu::Buffer, k: u32| {
            self.dispatch(
                "pack",
                &[input, &s["packed"]],
                [k / 32, b, 1],
                [k / 32, b, 7, 0],
            )
        };
        let add = || {
            self.dispatch(
                "add",
                &[&s["o"], &s["x"]],
                [(2048 * b + 255) / 256, 1, 1],
                [2048 * b, 0, 0, 0],
            )
        };
        for l in &self.layers {
            let m = &l.matrices;
            let n = &l.vectors;
            v.push(norm(&n["input_norm"]));
            v.push(pack(&s["z"], 2048));
            for name in ["q", "k", "v"] {
                v.push(self.project(&m[name], &s[name], b, false));
            }
            v.push(self.dispatch(
                "rope",
                &[
                    &s["q"],
                    &s["k"],
                    &s["v"],
                    &n["q_norm"],
                    &n["k_norm"],
                    &l.kv,
                    &self.vectors["freq"],
                    &s["state"],
                ],
                [24, b, 1],
                [self.kv_capacity, 0, 0, 0],
            ));
            v.push(self.dispatch(
                "attention",
                &[&s["q"], &l.kv, &s["att"], &s["state"]],
                [16, b, 1],
                [self.kv_capacity, 0, 0, 0],
            ));
            v.push(pack(&s["att"], 2048));
            v.push(self.project(&m["o"], &s["o"], b, false));
            v.push(add());
            v.push(norm(&n["post_norm"]));
            v.push(pack(&s["z"], 2048));
            for name in ["gate", "up"] {
                v.push(self.project(&m[name], &s[name], b, false));
            }
            v.push(self.dispatch(
                "silu",
                &[&s["gate"], &s["up"], &s["act"]],
                [(6144 * b + 255) / 256, 1, 1],
                [6144 * b, 0, 0, 0],
            ));
            v.push(pack(&s["act"], 6144));
            v.push(self.project(&m["down"], &s["o"], b, false));
            v.push(add());
        }
        v.push(norm(&self.vectors["norm"]));
        v.push(pack(&s["z"], 2048));
        v.push(self.project(&self.matrices["embed"], &s["logits"], b, true));
        v
    }
    async fn execute(&mut self, tokens: &[u32], full: bool, top_k: u32) -> Result<Vec<f32>> {
        let b = tokens.len() as u32;
        if b == 0
            || b > self.batch
            || self.pos + b > self.capacity
            || tokens.iter().any(|x| *x >= 151669)
        {
            return Err(err("invalid tokens or context exceeded"));
        }
        if self.layers.len() != 28 {
            return Err(err("model is not loaded"));
        }
        if self.pos + b > self.kv_capacity {
            let next = ((self.pos + b + 255) / 256 * 256).min(self.capacity);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            let old_size = self.kv_capacity as u64 * 1024 * 4;
            let new_size = next as u64 * 1024 * 4;
            for l in &mut self.layers {
                let kv = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("grown FP32 KV"),
                    size: new_size * 2,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(&l.kv, 0, &kv, 0, old_size);
                encoder.copy_buffer_to_buffer(&l.kv, old_size, &kv, new_size, old_size);
                l.kv = kv;
            }
            self.queue.submit(Some(encoder.finish()));
            self.kv_capacity = next;
            self.kv_bytes = new_size * 2 * 28;
            self.graphs.clear();
        }
        if !self.graphs.contains_key(&b) {
            self.graphs.insert(b, self.graph(b));
        }
        self.queue
            .write_buffer(&self.buffers["tokens"], 0, bytemuck::cast_slice(tokens));
        self.queue.write_buffer(
            &self.buffers["state"],
            0,
            bytemuck::cast_slice(&[self.pos, b, 0, 0]),
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Bonsai token/chunk"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("W1A4 graph"),
                timestamp_writes: None,
            });
            for d in &self.graphs[&b] {
                pass.set_pipeline(&self.kernels[&d.pipeline].pipeline);
                pass.set_bind_group(0, &d.group, &[]);
                pass.dispatch_workgroups(d.dims[0], d.dims[1], d.dims[2]);
            }
        }
        let (source, size) = if full {
            (&self.buffers["logits"], 151669 * 4u64)
        } else {
            let d = self.dispatch(
                "candidates",
                &[&self.buffers["logits"], &self.buffers["candidates"]],
                [64, 1, 1],
                [151669, top_k.clamp(1, 64), 0, 0],
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GPU top-k"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.kernels["candidates"].pipeline);
            pass.set_bind_group(0, &d.group, &[]);
            pass.dispatch_workgroups(64, 1, 1);
            drop(pass);
            (
                &self.buffers["candidates"],
                64 * top_k.clamp(1, 64) as u64 * 8,
            )
        };
        encoder.copy_buffer_to_buffer(source, 0, &self.buffers["readback"], 0, size);
        self.queue.submit(Some(encoder.finish()));
        let slice = self.buffers["readback"].slice(..size);
        let (tx, rx) = futures_channel::oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        rx.await.map_err(err)?.map_err(err)?;
        let result = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        self.buffers["readback"].unmap();
        self.pos += b;
        if full && result.iter().any(|x| !x.is_finite()) {
            return Err(err("non-finite logits"));
        }
        Ok(result)
    }
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(js_name=create)]
    pub async fn create(tokenizer_bytes: Vec<u8>, capacity: u32, batch: u32) -> Result<Engine> {
        console_error_panic_hook::set_once();
        if !(128..=4096).contains(&capacity) || !(1..=32).contains(&batch) {
            return Err(err("context 128..4096, batch 1..32"));
        }
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes).map_err(err)?;
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(err)?;
        let adapter_info = format!("{:?}", adapter.get_info());
        let limits = wgpu::Limits::default();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Bonsai W1A4"),
                required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(err)?;
        device.on_uncaptured_error(Box::new(|e| web_sys::console::error_1(&err(e))));
        let mut engine = Engine {
            device,
            queue,
            tokenizer,
            matrices: HashMap::new(),
            vectors: HashMap::new(),
            layers: Vec::new(),
            kernels: HashMap::new(),
            buffers: HashMap::new(),
            graphs: HashMap::new(),
            capacity,
            kv_capacity: capacity.min(256),
            batch,
            pos: 0,
            weights_bytes: 0,
            kv_bytes: 0,
            scratch_bytes: 0,
            adapter_info,
        };
        for (name, source) in [
            ("embed", include_str!("embed.wgsl")),
            ("project", include_str!("project.wgsl")),
            ("norm", include_str!("norm.wgsl")),
            ("pack", include_str!("pack.wgsl")),
            ("rope", include_str!("rope.wgsl")),
            ("attention", include_str!("attention.wgsl")),
            ("add", include_str!("add.wgsl")),
            ("silu", include_str!("silu.wgsl")),
            ("candidates", include_str!("candidates.wgsl")),
        ] {
            let source = source.replace("CONTEXT_LENGTH", &capacity.to_string());
            engine
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let module = engine
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            let pipeline =
                engine
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(name),
                        layout: None,
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });
            if let Some(e) = engine.device.pop_error_scope().await {
                return Err(err(format!("{name}: {e}")));
            }
            engine.kernels.insert(name.into(), Kernel { pipeline });
        }
        for (name, len) in [
            ("x", 2048 * batch),
            ("z", 2048 * batch),
            ("packed", 192 * 8 * batch),
            ("q", 2048 * batch),
            ("k", 1024 * batch),
            ("v", 1024 * batch),
            ("att", 2048 * batch),
            ("o", 2048 * batch),
            ("gate", 6144 * batch),
            ("up", 6144 * batch),
            ("act", 6144 * batch),
            ("logits", 151669),
            ("candidates", 64 * 64 * 2),
            ("state", 4),
            ("tokens", batch),
        ] {
            let buf = engine.buffer(
                name,
                len as u64 * 4,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            engine.scratch_bytes += len as u64 * 4;
            engine.buffers.insert(name.into(), buf);
        }
        let buf = engine.buffer(
            "readback",
            151669 * 4,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        engine.scratch_bytes += 151669 * 4;
        engine.buffers.insert("readback".into(), buf);
        Ok(engine)
    }
    pub fn adapter(&self) -> String {
        self.adapter_info.clone()
    }
    pub fn load_blob(&mut self, bytes: Vec<u8>, layer: i32) -> Result<()> {
        if bytes.len() < 8 {
            return Err(err("truncated blob"));
        }
        let h = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        if h > bytes.len() - 8 {
            return Err(err("invalid blob header"));
        }
        let blob: Blob = serde_json::from_slice(&bytes[8..8 + h]).map_err(err)?;
        let payload = &bytes[8 + h..];
        let get = |r: [usize; 2]| {
            payload
                .get(r[0]..r[1])
                .ok_or_else(|| err("invalid tensor range"))
        };
        let mut matrices = HashMap::new();
        let mut vectors = HashMap::new();
        for m in blob.matrices {
            if m.k % 128 != 0
                || get(m.w)?.len() != m.n as usize * m.k as usize / 8
                || get(m.s)?.len() != m.n as usize * m.k as usize / 128 * 2
            {
                return Err(err("invalid packed matrix shape"));
            }
            let scales: Vec<f32> = get(m.s)?
                .chunks_exact(2)
                .map(|x| half::f16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32())
                .collect();
            self.weights_bytes += (get(m.w)?.len() + scales.len() * 4) as u64;
            matrices.insert(
                m.name,
                Matrix {
                    weights: self.data("W1", get(m.w)?),
                    scales: self.data("scales", bytemuck::cast_slice(&scales)),
                    n: m.n,
                    k: m.k,
                },
            );
        }
        for n in blob.vectors {
            let data = get(n.data)?;
            self.weights_bytes += data.len() as u64;
            vectors.insert(n.name, self.data("norm/frequencies", data));
        }
        if layer == -1 {
            self.matrices = matrices;
            self.vectors = vectors;
        } else {
            if layer as usize != self.layers.len() || layer >= 28 {
                return Err(err("load layers in order"));
            }
            let size = self.kv_capacity as u64 * 1024 * 2 * 4;
            self.kv_bytes += size;
            let kv = self.buffer(
                "FP32 KV",
                size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            self.layers.push(Layer {
                matrices,
                vectors,
                kv,
            });
        }
        self.graphs.clear();
        Ok(())
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(text, false)
            .map_err(err)?
            .get_ids()
            .to_vec())
    }
    pub fn decode(&self, tokens: Vec<u32>) -> Result<String> {
        self.tokenizer.decode(&tokens, true).map_err(err)
    }
    pub fn reset(&mut self) {
        self.pos = 0;
    }
    pub fn truncate(&mut self, pos: u32) -> Result<()> {
        if pos > self.pos {
            return Err(err("invalid prefix"));
        }
        self.pos = pos;
        Ok(())
    }
    pub fn position(&self) -> u32 {
        self.pos
    }
    pub fn memory(&self) -> Result<JsValue> {
        let mem = wasm_bindgen::memory().unchecked_into::<js_sys::WebAssembly::Memory>();
        let bytes = js_sys::Uint8Array::new(&mem.buffer()).length() as u64;
        serde_wasm_bindgen::to_value(&Memory {
            gpu_weights_bytes: self.weights_bytes,
            gpu_kv_bytes: self.kv_bytes,
            gpu_scratch_bytes: self.scratch_bytes,
            wasm_linear_bytes: bytes,
            context: self.capacity,
            batch: self.batch,
            kv_capacity_tokens: self.kv_capacity,
            used_tokens: self.pos,
        })
        .map_err(err)
    }
    /// Returns [token-id-as-u32-bits, score] pairs from GPU partitions, globally sorted.
    pub async fn forward(&mut self, tokens: Vec<u32>, top_k: u32) -> Result<JsValue> {
        let data = self.execute(&tokens, false, top_k).await?;
        let mut candidates: Vec<(u32, f32)> = data
            .chunks_exact(2)
            .map(|p| (p[0].to_bits(), p[1]))
            .filter(|p| p.0 < 151669 && p.1.is_finite())
            .collect();
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
        candidates.truncate(top_k.clamp(1, 64) as usize);
        serde_wasm_bindgen::to_value(&candidates).map_err(err)
    }
    pub async fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.execute(&[token], true, 1).await
    }
    pub async fn logits_chunk(&mut self, tokens: Vec<u32>) -> Result<Vec<f32>> {
        self.execute(&tokens, true, 1).await
    }
    /// Real WebGPU pack and projection checks against a Rust scalar oracle.
    pub async fn kernel_check(&self) -> Result<JsValue> {
        if self.layers.len() != 28 {
            return Err(err("model not loaded"));
        }
        let m = &self.layers[0].matrices["q"];
        let words = self.read_buffer(&m.weights, (m.n * m.k / 8) as u64).await?;
        let scales: Vec<f32> = self
            .read_buffer(&m.scales, (m.n * m.k / 128 * 4) as u64)
            .await?
            .into_iter()
            .map(f32::from_bits)
            .collect();
        let mut rng = 123456789u32;
        let mut bit_errors = 0;
        let mut scale_errors = 0;
        let mut max_error = 0.0f32;
        let mut linear_errors = 0;
        for case in 0..40 {
            let input: Vec<f32> = (0..m.k)
                .map(|i| match case {
                    0 => 0.0,
                    1 => 7.0,
                    2 => -7.0,
                    3 => (i % 15) as f32 - 7.0,
                    4 => {
                        if i % 2 == 0 {
                            -7.0
                        } else {
                            7.0
                        }
                    }
                    5 => {
                        if i % 32 == 31 {
                            7.0
                        } else {
                            (i % 14) as f32 * 0.5 - 3.0
                        }
                    }
                    _ => {
                        rng ^= rng << 13;
                        rng ^= rng >> 17;
                        rng ^= rng << 5;
                        (rng as i32 as f32) / (i32::MAX as f32) * 4.0
                    }
                })
                .collect();
            self.queue
                .write_buffer(&self.buffers["z"], 0, bytemuck::cast_slice(&input));
            let pack = self.dispatch(
                "pack",
                &[&self.buffers["z"], &self.buffers["packed"]],
                [m.k / 32, 1, 1],
                [m.k / 32, 1, 7, 0],
            );
            let project = self.project(m, &self.buffers["q"], 1, false);
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                for d in [&pack, &project] {
                    pass.set_pipeline(&self.kernels[&d.pipeline].pipeline);
                    pass.set_bind_group(0, &d.group, &[]);
                    pass.dispatch_workgroups(d.dims[0], d.dims[1], d.dims[2]);
                }
            }
            self.queue.submit(Some(encoder.finish()));
            let actual = self
                .read_buffer(&self.buffers["packed"], (m.k / 32 * 8 * 4) as u64)
                .await?;
            let out = self.read_buffer(&self.buffers["q"], m.n as u64 * 4).await?;
            let groups = m.k as usize / 32;
            let mut p = vec![[0u32; 8]; groups];
            for g in 0..groups {
                let maximum = input[g * 32..g * 32 + 32]
                    .iter()
                    .copied()
                    .map(f32::abs)
                    .fold(0.0f32, f32::max);
                p[g][4] = (maximum / 7.0).to_bits();
                let (mut sum, mut neg) = (0i32, 0i32);
                for lane in 0..32 {
                    let q = if maximum == 0.0 {
                        0
                    } else {
                        ((input[g * 32 + lane] / maximum) * 7.0)
                            .round()
                            .clamp(-7.0, 7.0) as i32
                    };
                    let mag = q.unsigned_abs();
                    for b in 0..3 {
                        p[g][b] |= ((mag >> b) & 1) << lane;
                    }
                    if q < 0 {
                        p[g][3] |= 1 << lane;
                    }
                    sum += q;
                    neg += (-q).max(0);
                }
                p[g][5] = sum as u32;
                p[g][6] = neg as u32;
                for field in [0, 1, 2, 3, 5, 6] {
                    if p[g][field] != actual[g * 8 + field] {
                        bit_errors += 1;
                    }
                }
                if p[g][4] != actual[g * 8 + 4] {
                    scale_errors += 1;
                }
            }
            for row in 0..m.n as usize {
                let mut total = 0.0f32;
                for g in 0..groups {
                    let r = words[g * m.n as usize + row] ^ p[g][3];
                    let c = (r & p[g][0]).count_ones() as i32
                        + 2 * (r & p[g][1]).count_ones() as i32
                        + 4 * (r & p[g][2]).count_ones() as i32;
                    let s = scales[(g / 4) * m.n as usize + row];
                    let weighted = s * ((c - p[g][6] as i32) as f32);
                    let offset = (-s * 0.5) * (p[g][5] as i32 as f32);
                    total += (weighted + offset) * f32::from_bits(p[g][4]);
                }
                let error = (f32::from_bits(out[row]) - total).abs();
                max_error = max_error.max(error);
                if error > 2e-4 * (1.0 + total.abs()) {
                    linear_errors += 1;
                }
            }
        }
        #[derive(Serialize)]
        struct Check {
            cases: u32,
            linear_outputs: u32,
            bit_errors: u32,
            scale_errors: u32,
            linear_errors: u32,
            max_abs_error: f32,
        }
        serde_wasm_bindgen::to_value(&Check {
            cases: 40,
            linear_outputs: 40 * m.n,
            bit_errors,
            scale_errors,
            linear_errors,
            max_abs_error: max_error,
        })
        .map_err(err)
    }
    /// Instrumented first layer for cross-backend operator comparison.
    /// Developer-only dispatch profile. Separate passes perturb timings, so
    /// use these to locate expensive kernels, not as end-to-end benchmarks.
    pub async fn profile(&mut self, token: u32, position: u32) -> Result<JsValue> {
        if !self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Err(err("GPU timestamp queries unavailable"));
        }
        if position >= self.kv_capacity { return Err(err("profile position exceeds allocated KV")); }
        self.queue.write_buffer(&self.buffers["tokens"],0,bytemuck::cast_slice(&[token]));
        self.queue.write_buffer(&self.buffers["state"],0,bytemuck::cast_slice(&[position,1,0,0]));
        let graph=self.graph(1);
        let queries=self.device.create_query_set(&wgpu::QuerySetDescriptor{label:Some("dispatch profile"),ty:wgpu::QueryType::Timestamp,count:graph.len() as u32*2});
        let resolved=self.buffer("timestamps",graph.len() as u64*16,wgpu::BufferUsages::QUERY_RESOLVE|wgpu::BufferUsages::COPY_SRC);
        let mut encoder=self.device.create_command_encoder(&Default::default());
        for (i,d) in graph.iter().enumerate() {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{label:Some(&d.pipeline),timestamp_writes:Some(wgpu::ComputePassTimestampWrites{query_set:&queries,beginning_of_pass_write_index:Some(i as u32*2),end_of_pass_write_index:Some(i as u32*2+1)})});
            pass.set_pipeline(&self.kernels[&d.pipeline].pipeline);
            pass.set_bind_group(0,&d.group,&[]);
            pass.dispatch_workgroups(d.dims[0],d.dims[1],d.dims[2]);
        }
        encoder.resolve_query_set(&queries,0..graph.len() as u32*2,&resolved,0);
        self.queue.submit(Some(encoder.finish()));
        let data=self.read_buffer(&resolved,graph.len() as u64*16).await?;
        let mut totals:HashMap<String,f64>=HashMap::new();
        let mut rows=Vec::new();
        let period=self.queue.get_timestamp_period() as f64;
        for (i,d) in graph.iter().enumerate() {
            let a=data[i*4] as u64|((data[i*4+1] as u64)<<32);
            let b=data[i*4+2] as u64|((data[i*4+3] as u64)<<32);
            let us=(b.saturating_sub(a)) as f64*period/1000.0;
            *totals.entry(d.pipeline.clone()).or_default()+=us;
            rows.push(serde_json::json!({"kernel":d.pipeline,"us":us,"index":i}));
        }
        serde_json::json!({"dispatches":graph.len(),"totals_us":totals,"rows":rows}).serialize(&serde_wasm_bindgen::Serializer::json_compatible()).map_err(err)
    }
    pub async fn trace(&mut self, token: u32) -> Result<JsValue> {
        if self.layers.len() != 28 {
            return Err(err("model not loaded"));
        }
        self.queue
            .write_buffer(&self.buffers["tokens"], 0, bytemuck::cast_slice(&[token]));
        self.queue.write_buffer(
            &self.buffers["state"],
            0,
            bytemuck::cast_slice(&[self.pos, 1, 0, 0]),
        );
        let graph = self.graph(1);
        let mut snapshots = Vec::new();
        let fields = [
            ("embed", "x", 2048),
            ("norm1", "z", 2048),
            ("pack1", "packed", 64 * 8),
            ("q", "q", 2048),
            ("k", "k", 1024),
            ("v", "v", 1024),
            ("rope_q", "q", 2048),
            ("attention", "att", 2048),
            ("pack_att", "packed", 64 * 8),
            ("o", "o", 2048),
            ("residual1", "x", 2048),
            ("norm2", "z", 2048),
            ("pack2", "packed", 64 * 8),
            ("gate", "gate", 6144),
            ("up", "up", 6144),
            ("silu", "act", 6144),
            ("pack_down", "packed", 192 * 8),
            ("down", "o", 2048),
            ("residual2", "x", 2048),
        ];
        for (index, d) in graph.iter().take(19).enumerate() {
            let (name, field, len) = fields[index];
            let size = len as u64 * 4;
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.kernels[&d.pipeline].pipeline);
                pass.set_bind_group(0, &d.group, &[]);
                pass.dispatch_workgroups(d.dims[0], d.dims[1], d.dims[2]);
            }
            encoder.copy_buffer_to_buffer(
                &self.buffers[field],
                0,
                &self.buffers["readback"],
                0,
                size,
            );
            self.queue.submit(Some(encoder.finish()));
            let slice = self.buffers["readback"].slice(..size);
            let (tx, rx) = futures_channel::oneshot::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            rx.await.map_err(err)?.map_err(err)?;
            let value = if field == "packed" {
                serde_json::json!(
                    bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec()
                )
            } else {
                serde_json::json!(
                    bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec()
                )
            };
            self.buffers["readback"].unmap();
            snapshots.push(serde_json::json!({"name":name,"values":value}));
        }
        snapshots
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(err)
    }
}
