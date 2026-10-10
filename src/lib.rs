//! Dedicated Bonsai W1A4 browser executor. No native service or generic ML runtime.
mod kv;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::cell::RefCell;
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;

fn err(s: impl ToString) -> JsValue {
    JsValue::from_str(&s.to_string())
}
type Result<T> = std::result::Result<T, JsValue>;
fn load_tokenizer(bytes: &[u8]) -> Result<Tokenizer> {
    if !bytes.starts_with(b"BNSITOK1") { return Tokenizer::from_bytes(bytes).map_err(err); }
    let mut cursor=8usize;
    fn word(bytes:&[u8],cursor:&mut usize)->Result<usize> {
        let b=bytes.get(*cursor..*cursor+4).ok_or_else(||err("truncated tokenizer"))?;
        *cursor+=4;Ok(u32::from_le_bytes(b.try_into().unwrap()) as usize)
    }
    let len=word(bytes,&mut cursor)?;
    let meta=bytes.get(cursor..cursor+len).ok_or_else(||err("invalid tokenizer metadata"))?;
    cursor+=len;
    let mut config:serde_json::Value=serde_json::from_slice(meta).map_err(err)?;
    let added:Vec<tokenizers::AddedToken>=serde_json::from_value(config["added_tokens"].clone()).map_err(err)?;
    config["added_tokens"]=serde_json::json!([]);
    let mut tokenizer=Tokenizer::from_bytes(serde_json::to_vec(&config).map_err(err)?).map_err(err)?;
    let count=word(bytes,&mut cursor)?;
    if count!=151643 { return Err(err("unexpected Bonsai vocabulary")); }
    let mut vocab=ahash::AHashMap::with_capacity(count);
    let mut tokens=Vec::with_capacity(count);
    for id in 0..count {
        let len=word(bytes,&mut cursor)?;
        let s=std::str::from_utf8(bytes.get(cursor..cursor+len).ok_or_else(||err("truncated token"))?).map_err(err)?.to_string();cursor+=len;
        vocab.insert(s.clone(),id as u32);tokens.push(s);
    }
    let count=word(bytes,&mut cursor)?;let mut merges=Vec::with_capacity(count);
    for _ in 0..count {
        let a=word(bytes,&mut cursor)?;let b=word(bytes,&mut cursor)?;
        merges.push((tokens.get(a).ok_or_else(||err("invalid merge"))?.clone(),tokens.get(b).ok_or_else(||err("invalid merge"))?.clone()));
    }
    if cursor!=bytes.len(){return Err(err("trailing tokenizer data"));}
    let model=tokenizers::models::bpe::BPE::builder().vocab_and_merges(vocab,merges)
        .continuing_subword_prefix(String::new()).end_of_word_suffix(String::new()).build().map_err(err)?;
    tokenizer.with_model(model);
    tokenizer.add_tokens(&added);
    Ok(tokenizer)
}


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
    handle: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    dims: [u32; 3],
}
struct Layer {
    matrices: HashMap<String, Matrix>,
    vectors: HashMap<String, wgpu::Buffer>,
    kv: wgpu::Buffer,
}
struct KvPrefix {
    tokens: u32,
    bits: u32,
    layers: Vec<wgpu::Buffer>,
}
#[derive(Serialize)]
struct Memory {
    gpu_weights_bytes: u64,
    gpu_kv_bytes: u64,
    gpu_prefix_bytes: u64,
    gpu_scratch_bytes: u64,
    wasm_linear_bytes: u64,
    context: u32,
    batch: u32,
    kv_capacity_tokens: u32,
    used_tokens: u32,
    kv_bits: u32,
    gpu_kv_full_context_bytes: u64,
    yarn_factor: f32,
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
    candidates: HashMap<u32, Dispatch>,
    uniforms: RefCell<HashMap<[u32;4],wgpu::Buffer>>,
    capacity: u32,
    kv_capacity: u32,
    kv_bits: u32,
    prefixes: HashMap<u32, KvPrefix>,
    short_attention: u32,
    force_tiled: bool,
    sparse_prefill: bool,
    prefill_end: u32,
    sparse_alpha: f32,
    batch: u32,
    pos: u32,
    weights_bytes: u64,
    kv_bytes: u64,
    scratch_bytes: u64,
    adapter_info: String,
    fuse_qkv: bool,
    optimized: bool,
    unroll_projection: bool,
}

impl Engine {
    fn ensure_kv_capacity(&mut self, needed: u32) -> Result<()> {
        if needed>self.capacity {return Err(err("KV capacity exceeded"));}
        if needed<=self.kv_capacity {return Ok(());}
        let next=(self.kv_capacity*2).max((needed+255)/256*256).min(self.capacity);
        let width=if self.kv_bits==4 {160u32}else{1024u32};
        let mut encoder=self.device.create_command_encoder(&Default::default());
        let new_size=next as u64*width as u64*4;
        let mut growth=Vec::new();
        for l in &self.layers {
            let kv=self.buffer("grown KV",new_size*2,wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST);
            let groups=(self.kv_capacity*width*2+255)/256;
            let d=self.dispatch("kv_grow",&[&l.kv,&kv],[groups.min(65535),(groups+65534)/65535,1],[self.kv_capacity,next,if self.kv_bits==4 {5}else{1},width]);
            growth.push((kv,d));
        }
        {let mut pass=encoder.begin_compute_pass(&Default::default());
         for (_,d) in &growth {pass.set_pipeline(&d.handle);pass.set_bind_group(0,&d.group,&[]);pass.dispatch_workgroups(d.dims[0],d.dims[1],1);}}
        self.queue.submit(Some(encoder.finish()));
        for(l,(kv,_))in self.layers.iter_mut().zip(growth){l.kv=kv;}
        self.kv_capacity=next;self.kv_bytes=new_size*2*self.layers.len() as u64;self.graphs.clear();Ok(())
    }
    fn prefix_size(&self,tokens:u32)->u64 {tokens as u64*self.layers.len() as u64*if self.kv_bits==4 {1280}else{8192}}
    fn check_prefix(&self,slot:u32,tokens:u32)->Result<()> {
        if slot>=2||tokens==0||tokens>2048||tokens>self.capacity||self.layers.len()!=28||self.prefix_size(tokens)>64*1024*1024 {return Err(err("invalid or oversized system prefix"));}
        Ok(())
    }
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
    fn data_js(&self,label:&str,bytes:&js_sys::Uint8Array)->wgpu::Buffer {
        let buffer=self.device.create_buffer(&wgpu::BufferDescriptor{label:Some(label),size:(bytes.length() as u64+3)&!3,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::COPY_SRC,mapped_at_creation:true});
        // The WebGPU mapping already lives in JavaScript. Copy directly into
        // that view instead of round-tripping a zero-filled mapping via WASM.
        #[cfg(target_arch="wasm32")]
        {let view=buffer.slice(..).get_mapped_range();view.as_uint8array().set(bytes,0);}
        #[cfg(not(target_arch="wasm32"))]
        {let mut view=buffer.slice(..).get_mapped_range_mut();view.copy_from_slice(&bytes.to_vec());}
        buffer.unmap();buffer
    }
    fn uniform(&self, data: &[u32]) -> wgpu::Buffer {
        let key:[u32;4]=data.try_into().unwrap();
        let mut uniforms=self.uniforms.borrow_mut();
        uniforms.entry(key).or_insert_with(||self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("shared shape"),contents:bytemuck::cast_slice(data),usage:wgpu::BufferUsages::UNIFORM})).clone()
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
            handle: p.clone(),
            group,
            dims,
        }
    }
    fn project(&self, m: &Matrix, out: &wgpu::Buffer, b: u32, tail: bool) -> Dispatch {
        let count = if tail { 1 } else { b };
        self.dispatch(
            if self.unroll_projection && self.optimized {"project"}else{"project_exact"},
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
    fn compile_kernel(&mut self,name:&str,source:&str) {
            let source = source.replace("CONTEXT_LENGTH", &self.short_attention.to_string()).replace("YARN_ATTN", &(1.0+0.1*(if self.capacity>32768 {16.0f32}else{4.0f32}).ln()).to_string());
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                });
            let pipeline =
                self
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(name),
                        layout: None,
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: Default::default(),
                        cache: None,
                    });
            self.kernels.insert(name.into(), Kernel { pipeline });
    }
    async fn prepare_reference(&mut self)->Result<()> {
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        for (name,source) in [
            ("project_exact",include_str!("project_exact.wgsl")),
            ("qkv_exact",include_str!("qkv_exact.wgsl")),
            ("gateup_exact",include_str!("gateup_exact.wgsl")),
            ("attention",include_str!("attention.wgsl")),
            ("add",include_str!("add.wgsl")),
            ("silu",include_str!("silu.wgsl")),
        ] {
            if name=="qkv_exact" && !self.fuse_qkv {continue;}
            if !self.kernels.contains_key(name) {self.compile_kernel(name,source);}
        }
        if let Some(e)=self.device.pop_error_scope().await {return Err(err(format!("reference pipelines: {e}")));}
        Ok(())
    }
    fn graph(&self,b:u32)->Vec<Dispatch>{self.graph_mode(b,self.optimized,false)}
    fn graph_mode(&self, b: u32, optimized:bool, all_logits:bool) -> Vec<Dispatch> {
        let sparse=self.sparse_prefill&&self.kv_bits==4&&b>1&&self.pos>=8192&&self.pos/128+2<(self.prefill_end+127)/128;
        let tiled=sparse||self.force_tiled||self.pos+b>self.short_attention;
        let parts=(self.pos+b+127)/128;
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
        for (index,l) in self.layers.iter().enumerate() {
            let m = &l.matrices;
            let n = &l.vectors;
            if optimized && index>0 {v.push(self.dispatch("norm_add",&[&s["x"],&n["input_norm"],&s["z"],&s["o"]],[1,b,1],[2048,b,0,0]));}else{v.push(norm(&n["input_norm"]));}
            v.push(pack(&s["z"], 2048));
            if optimized && self.fuse_qkv {
                v.push(self.dispatch(if self.unroll_projection {"qkv"}else{"qkv_exact"},&[&m["q"].weights,&m["q"].scales,&s["q"],&m["k"].weights,&m["k"].scales,&s["k"],&m["v"].weights,&m["v"].scales,&s["v"],&s["packed"]],[32,b,1],[0,64,b,0]));
            } else {for name in ["q","k","v"]{v.push(self.project(&m[name],&s[name],b,false));}}
            v.push(self.dispatch(
                if self.kv_bits==4 {"rope_q4"}else{"rope"},
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
            if sparse {
                v.push(self.dispatch("flash_pool_q4",&[&l.kv,&s["pooled"],&s["state"]],[8,parts,1],[self.kv_capacity,parts,if sparse {1}else{0},0]));
                v.push(self.dispatch("flash_select",&[&s["q"],&s["pooled"],&s["selected"],&s["state"]],[16,1,1],[0,parts,0,self.sparse_alpha.to_bits()]));
            }
            if tiled {
                v.push(self.dispatch(if self.kv_bits==4 {"attention_partial_q4"}else{"attention_partial"},&[&s["q"],&l.kv,&s["parts"],&s["state"],&s["selected"]],[16,b,parts],[self.kv_capacity,parts,if sparse {1}else{0},0]));
                v.push(self.dispatch("attention_merge",&[&s["parts"],&s["att"]],[16,b,1],[parts,0,0,0]));
                v.push(pack(&s["att"],2048));
            } else if optimized {
                v.push(self.dispatch(if self.kv_bits==4 {"attention_pack_q4"}else{"attention_pack"},&[&s["q"],&l.kv,&s["packed"],&s["state"]],[16,b,1],[self.kv_capacity,0,0,0]));
            } else {
                v.push(self.dispatch(if self.kv_bits==4 {"attention_q4"}else{"attention"},&[&s["q"],&l.kv,&s["att"],&s["state"]],[16,b,1],[self.kv_capacity,0,0,0]));
                v.push(pack(&s["att"],2048));
            }
            v.push(self.project(&m["o"], &s["o"], b, false));
            if optimized {v.push(self.dispatch("norm_add",&[&s["x"],&n["post_norm"],&s["z"],&s["o"]],[1,b,1],[2048,b,0,0]));}
            else {v.push(add());v.push(norm(&n["post_norm"]));}
            v.push(pack(&s["z"], 2048));
            if optimized {
                v.push(self.dispatch(if self.unroll_projection {"gateup"}else{"gateup_exact"},&[&m["gate"].weights,&m["gate"].scales,&s["gate"],&m["up"].weights,&m["up"].scales,&s["up"],&s["packed"]],[96,b,1],[0,64,b,0]));
                v.push(self.dispatch("silu_pack",&[&s["gate"],&s["up"],&s["packed"]],[192,b,1],[192,b,7,0]));
            } else {
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
            }
            v.push(self.project(&m["down"], &s["o"], b, false));
            if !optimized {v.push(add());}
        }
        if optimized {v.push(self.dispatch("norm_add",&[&s["x"],&self.vectors["norm"],&s["z"],&s["o"]],[1,b,1],[2048,b,0,0]));}else{v.push(norm(&self.vectors["norm"]));}
        v.push(pack(&s["z"], 2048));
        v.push(self.project(&self.matrices["embed"], &s["logits"], b, !all_logits));
        v
    }
    async fn execute(&mut self, tokens: &[u32], full: bool, top_k: u32, all: bool) -> Result<Vec<f32>> {
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
        self.ensure_kv_capacity(self.pos+b)?;
        let sparse=self.sparse_prefill&&self.kv_bits==4&&b>1&&self.pos>=8192&&self.pos/128+2<(self.prefill_end+127)/128;
        let graph_key=b+if sparse {512}else{0}+if all{256}else{0}+if self.force_tiled||self.pos+b>self.short_attention {((self.pos+b+127)/128)*1024}else{0};
        // A tiled graph depends on the active split count. Keep only the current
        // split to bound bind groups during a 128K prefill.
        if graph_key>=1024 {self.graphs.retain(|key,_|*key<1024||*key/1024==graph_key/1024);}
        if !self.graphs.contains_key(&graph_key) {
            self.graphs.insert(graph_key, self.graph_mode(b,self.optimized,all));
        }
        self.queue
            .write_buffer(&self.buffers["tokens"], 0, bytemuck::cast_slice(tokens));
        self.queue.write_buffer(
            &self.buffers["state"],
            0,
            bytemuck::cast_slice(&[self.pos, b, 0, 0]),
        );
        let top_k=top_k.clamp(1,64);
        let count=if all{b}else{1};let candidate_key=top_k+count*256;
        if !full && !self.candidates.contains_key(&candidate_key) {
            let d=self.dispatch(if top_k==1{"candidates"}else{"candidates_cached"},&[&self.buffers["logits"],&self.buffers["candidates"]],[64,count,1],[151669,top_k,0,0]);
            self.candidates.insert(candidate_key,d);
        }
        let mut encoder=self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("Bonsai token/chunk")});
        {
            let mut pass=encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{label:Some("W1A4 graph"),timestamp_writes:None});
            for d in &self.graphs[&graph_key] {
                pass.set_pipeline(&d.handle);
                pass.set_bind_group(0,&d.group,&[]);
                pass.dispatch_workgroups(d.dims[0],d.dims[1],d.dims[2]);
            }
            if !full {
                let d=&self.candidates[&candidate_key];
                pass.set_pipeline(&d.handle);
                pass.set_bind_group(0,&d.group,&[]);
                pass.dispatch_workgroups(64,count,1);
            }
        }
        let (source,size)=if full{(&self.buffers["logits"],151669*4u64)}else{(&self.buffers["candidates"],64*top_k as u64*8*count as u64)};
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
    pub async fn create(tokenizer_bytes: Vec<u8>, capacity: u32, batch: u32, kv_bits: Option<u32>) -> Result<Engine> {
        console_error_panic_hook::set_once();
        let kv_bits=kv_bits.unwrap_or(4);
        if !(128..=131072).contains(&capacity) || !(1..=32).contains(&batch) || !matches!(kv_bits,4|32) {
            return Err(err("context 128..131072, batch 1..32, kv_bits 4 or 32"));
        }
        let tokenizer = load_tokenizer(&tokenizer_bytes)?;
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
        let fuse_qkv=adapter.limits().max_storage_buffers_per_shader_stage>=10;
        let mut limits = wgpu::Limits::default();
        // attention_pack uses four bytes per context token plus 4608 fixed bytes.
        // WebGPU's default limit is 16 KiB even on adapters offering 32 KiB.
        limits.max_compute_workgroup_storage_size=adapter.limits().max_compute_workgroup_storage_size.min(32768);
        let supported=((limits.max_compute_workgroup_storage_size.saturating_sub(4608))/4/128)*128;
        let short_attention=capacity.min(supported).min(4096);
        let kv_size=capacity as u64*if kv_bits==4 {1280}else{8192};
        if kv_size>adapter.limits().max_storage_buffer_binding_size as u64||kv_size>adapter.limits().max_buffer_size {return Err(err("当前 GPU 的单缓冲区上限不足以支持所选上下文；请选择更短的上下文。"));}
        limits.max_storage_buffer_binding_size=limits.max_storage_buffer_binding_size.max(kv_size as u32);
        limits.max_buffer_size=limits.max_buffer_size.max(kv_size);
        if fuse_qkv { limits.max_storage_buffers_per_shader_stage=10; }
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
            candidates: HashMap::new(),
            uniforms: RefCell::new(HashMap::new()),
            capacity,
            kv_capacity: capacity.min(256),
            kv_bits,prefixes:HashMap::new(),short_attention,force_tiled:false,sparse_prefill:false,prefill_end:0,sparse_alpha:0.1,
            batch,
            pos: 0,
            weights_bytes: 0,
            kv_bytes: 0,
            scratch_bytes: 0,
            adapter_info,
            fuse_qkv,
            optimized:true,
            unroll_projection: true,
        };
        engine.device.push_error_scope(wgpu::ErrorFilter::Validation);
        for (name, source) in [
            ("embed", include_str!("embed.wgsl")),
            ("expand_scales",include_str!("expand_scales.wgsl")),
            ("project", include_str!("project.wgsl")),
            ("project_exact",include_str!("project_exact.wgsl")),
            ("qkv_exact",include_str!("qkv_exact.wgsl")),
            ("gateup_exact",include_str!("gateup_exact.wgsl")),
            ("norm", include_str!("norm.wgsl")),
            ("pack", include_str!("pack.wgsl")),
            ("rope", include_str!("rope.wgsl")),
            ("attention", include_str!("attention.wgsl")),
            ("add", include_str!("add.wgsl")),
            ("silu", include_str!("silu.wgsl")),
            ("candidates", include_str!("candidates.wgsl")),
            ("qkv",include_str!("qkv.wgsl")),
            ("gateup",include_str!("gateup.wgsl")),
            ("norm_add",include_str!("norm_add.wgsl")),
            ("silu_pack",include_str!("silu_pack.wgsl")),
            ("kv_grow",include_str!("kv_grow.wgsl")),
            ("kv_copy",include_str!("kv_copy.wgsl")),
            ("attention_pack",include_str!("attention_pack.wgsl")),
            ("rope_q4",include_str!("rope_q4.wgsl")),
            ("attention_q4",include_str!("attention_q4.wgsl")),
            ("attention_pack_q4",include_str!("attention_pack_q4.wgsl")),
            ("attention_partial",include_str!("attention_partial.wgsl")),
            ("attention_partial_q4",include_str!("attention_partial_q4.wgsl")),
            ("attention_merge",include_str!("attention_merge.wgsl")),
            ("flash_pool_q4",include_str!("flash_pool_q4.wgsl")),
            ("flash_select",include_str!("flash_select.wgsl")),
            ("candidates_cached",include_str!("candidates_cached.wgsl")),
        ] {
            if (name=="qkv" || name=="qkv_exact") && !fuse_qkv { continue; }
            if matches!(name,"project_exact"|"qkv_exact"|"gateup_exact"|"attention"|"add"|"silu") {continue;}
            engine.compile_kernel(name,source);
        }
        if let Some(e)=engine.device.pop_error_scope().await { return Err(err(format!("pipeline initialization: {e}"))); }
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
            ("logits", 151669 * batch),
            ("candidates", 64 * 64 * 2 * batch),
            ("state", 4),
            ("tokens", batch),
            ("parts", ((capacity+127)/128)*batch*16*130),
            ("selected", 16*((capacity+127)/128+1)),
            ("pooled", 8*((capacity+127)/128)*128),
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
    pub fn load_blob(&mut self, bytes: js_sys::Uint8Array, layer: i32) -> Result<()> {
        if bytes.length() < 8 {
            return Err(err("truncated blob"));
        }
        let h = u64::from_le_bytes(bytes.subarray(0,8).to_vec().try_into().unwrap()) as usize;
        if h > bytes.length() as usize - 8 {
            return Err(err("invalid blob header"));
        }
        let blob: Blob = serde_json::from_slice(&bytes.subarray(8,(8+h) as u32).to_vec()).map_err(err)?;
        let get=|r:[usize;2]|->Result<js_sys::Uint8Array>{
            if r[0]>r[1] || r[1]>bytes.length() as usize-8-h {return Err(err("invalid tensor range"));}
            Ok(bytes.subarray((8+h+r[0]) as u32,(8+h+r[1]) as u32))
        };
        let mut matrices = HashMap::new();
        let mut vectors = HashMap::new();
        let mut expanded=Vec::new();
        for m in blob.matrices {
            if m.k % 128 != 0
                || get(m.w)?.length() as usize != m.n as usize * m.k as usize / 8
                || get(m.s)?.length() as usize != m.n as usize * m.k as usize / 128 * 2
            {
                return Err(err("invalid packed matrix shape"));
            }
            let source=self.data_js("packed FP16 scales",&get(m.s)?);
            let scales=self.buffer("FP32 scales",get(m.s)?.length() as u64*2,wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC);
            let count=get(m.s)?.length()/4;
            let expansion=self.dispatch("expand_scales",&[&source,&scales],[(count+255)/256,1,1],[count,0,0,0]);
            expanded.push((source,expansion));
            self.weights_bytes += (get(m.w)?.length() + get(m.s)?.length()*2) as u64;
            matrices.insert(
                m.name,
                Matrix {
                    weights: self.data_js("W1", &get(m.w)?),
                    scales,
                    n: m.n,
                    k: m.k,
                },
            );
        }
        let mut encoder=self.device.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());for (_,d) in &expanded {pass.set_pipeline(&d.handle);pass.set_bind_group(0,&d.group,&[]);pass.dispatch_workgroups(d.dims[0],d.dims[1],1);}}
        self.queue.submit(Some(encoder.finish()));
        for n in blob.vectors {
            let data = get(n.data)?;
            self.weights_bytes += data.length() as u64;
            if n.name=="freq"&&self.capacity>32768 {
                let low=(128.0f64*(8192.0/(32.0*std::f64::consts::TAU)).ln()/(2.0*1_000_000.0f64.ln())).floor();
                let high=(128.0f64*(8192.0/std::f64::consts::TAU).ln()/(2.0*1_000_000.0f64.ln())).ceil();
                let freq:Vec<f32>=(0..64).map(|i|{let inv=1_000_000.0f64.powf(-(2*i) as f64/128.0);let ramp=((i as f64-low)/(high-low)).clamp(0.0,1.0);(inv/16.0*ramp+inv*(1.0-ramp)) as f32}).collect();
                vectors.insert(n.name,self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("experimental YaRN x16"),contents:bytemuck::cast_slice(&freq),usage:wgpu::BufferUsages::STORAGE}));
            }else{vectors.insert(n.name, self.data_js("norm/frequencies", &data));}
        }
        if layer == -1 {
            self.matrices = matrices;
            self.vectors = vectors;
        } else {
            if layer as usize != self.layers.len() || layer >= 28 {
                return Err(err("load layers in order"));
            }
            let size = self.kv_capacity as u64 * if self.kv_bits==4 {1280}else{8192};
            self.kv_bytes += size;
            let kv = self.buffer(
                "KV cache",
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
    pub async fn finish_load(&self)->Result<()> {
        if self.layers.len()!=28 || !self.matrices.contains_key("embed") {return Err(err("model is not loaded"));}
        let (tx,rx)=futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {let _=tx.send(());});
        rx.await.map_err(err)?;Ok(())
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
    pub async fn set_reference(&mut self, reference:bool)->Result<()>{if reference {self.prepare_reference().await?;}self.optimized=!reference;self.graphs.clear();Ok(())}
    pub async fn set_unroll_projection(&mut self, unroll:bool)->Result<()>{if !unroll {self.prepare_reference().await?;}self.unroll_projection=unroll;self.graphs.clear();Ok(())}
    /// Synthetic real-GPU verification of KV packing and split attention.
    /// Long positions do not imply a 128K language-model quality evaluation.
    pub async fn kv_check(&self,count:u32)->Result<JsValue>{
        if self.kv_bits!=4||count==0||count>self.capacity{return Err(err("kv_check requires INT4 KV and count <= context"));}
        let data=|name:&str,values:&[u32]|self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some(name),contents:bytemuck::cast_slice(values),usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC});
        let b=self.batch.min(4);let cap=128u32;
        let kval:Vec<f32>=(0..b*1024).map(|i|if i<1024 {0.0}else{((i*97%997)as f32-498.0)/199.0}).collect();
        let vval:Vec<f32>=(0..b*1024).map(|i|((i*31%991)as f32-495.0)/99.0).collect();
        let k=data("check k",bytemuck::cast_slice(&kval));let v=data("check v",bytemuck::cast_slice(&vval));
        let q=self.buffer("check q",b as u64*2048*4,wgpu::BufferUsages::STORAGE);
        let norm=data("check norm",&vec![1.0f32.to_bits();128]);let freq=data("check freq",&vec![0;64]);let state=data("check state",&[0,b,0,0]);
        let packed=self.buffer("check packed KV",cap as u64*1280,wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC);
        let d=self.dispatch("rope_q4",&[&q,&k,&v,&norm,&norm,&packed,&freq,&state],[24,b,1],[cap,0,0,0]);
        let mut encoder=self.device.create_command_encoder(&Default::default());{let mut pass=encoder.begin_compute_pass(&Default::default());pass.set_pipeline(&d.handle);pass.set_bind_group(0,&d.group,&[]);pass.dispatch_workgroups(24,b,1);}self.queue.submit(Some(encoder.finish()));
        let actual=self.read_buffer(&packed,cap as u64*1280).await?;let factor=1.0+0.1*(if self.capacity>32768 {16.0f32}else{4.0f32}).ln();let mut errors=0u32;let mut pack_max=0.0f32;
        for row in 0..b as usize{for head in 0..8usize{
            let raw=&kval[row*1024+head*128..row*1024+head*128+128];let mut sums:Vec<f32>=raw.iter().map(|x|x*x).collect();for step in [64,32,16,8,4,2,1]{for i in 0..step{sums[i]+=sums[i+step];}}
            let inv=(sums[0]/128.0+1e-6).sqrt().recip();
            for dim in 0..128usize{let g=head*4+dim/32;let ko=(g*cap as usize+row)*5;let vo=cap as usize*160+row*160+g*5;
                for (offset,expected) in [(ko,raw[dim]*inv*factor),(vo,vval[row*1024+head*128+dim])]{let decoded=kv::decode(&actual[offset..offset+5],dim%32);let error=(decoded-expected).abs();pack_max=pack_max.max(error);let scale=f32::from_bits(actual[offset+4]);if !decoded.is_finite()||error>scale*0.5001+2e-5*(1.0+expected.abs()){errors+=1;}}
            }
        }}
        let count=count as usize;let parts=(count+127)/128;let mut bytes=vec![0u32;count*320];let query:Vec<f32>=(0..2048).map(|i|((i*11%61)as f32-30.0)/97.0).collect();let mut logits=Vec::with_capacity(count);
        for t in 0..count{let mut dot=0.0f32;for group in 0..4{let mut a=[0.;32];let mut v=[0.;32];for j in 0..32{a[j]=(((t*17+(group*32+j)*31)%101)as f32-50.0)/37.0;v[j]=(((t*13+(group*32+j)*7)%97)as f32-48.0)/41.0;}
            let kp=kv::encode(&a);let vp=kv::encode(&v);let ko=(group*count+t)*5;let vo=count*160+t*160+group*5;bytes[ko..ko+5].copy_from_slice(&kp);bytes[vo..vo+5].copy_from_slice(&vp);for j in 0..32{dot+=query[group*32+j]*kv::decode(&kp,j);}}
            logits.push((dot*0.08838834764831845)as f64);
        }
        let kv=data("long check KV",&bytes);let query=data("long check query",bytemuck::cast_slice(&query));let state=data("long check state",&[(count-1)as u32,1,0,0]);let selected=data("unused selected",&[0;16]);
        let partial=self.buffer("long check partial",parts as u64*16*130*4,wgpu::BufferUsages::STORAGE);let out=self.buffer("long check output",2048*4,wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC);
        let a=self.dispatch("attention_partial_q4",&[&query,&kv,&partial,&state,&selected],[1,1,parts as u32],[count as u32,parts as u32,0,0]);let m=self.dispatch("attention_merge",&[&partial,&out],[1,1,1],[parts as u32,0,0,0]);
        let start=js_sys::Date::now();let mut encoder=self.device.create_command_encoder(&Default::default());{let mut pass=encoder.begin_compute_pass(&Default::default());for d in [&a,&m]{pass.set_pipeline(&d.handle);pass.set_bind_group(0,&d.group,&[]);pass.dispatch_workgroups(d.dims[0],d.dims[1],d.dims[2]);}}self.queue.submit(Some(encoder.finish()));
        let values=self.read_buffer(&out,128*4).await?;let elapsed=js_sys::Date::now()-start;let maximum=logits.iter().copied().fold(f64::NEG_INFINITY,f64::max);let weights:Vec<f64>=logits.iter().map(|x|(x-maximum).exp()).collect();let total: f64=weights.iter().sum();let mut expected=[0f64;128];
        for t in 0..count{for d in 0..128{let base=count*160+t*160+(d/32)*5;expected[d]+=weights[t]*kv::decode(&bytes[base..base+5],d%32)as f64/total;}}
        let max_error=(0..128).map(|i|(f32::from_bits(values[i])as f64-expected[i]).abs()).fold(0f64,f64::max);
        Ok(serde_json::json!({"tokens":count,"packing_values":b*1024*2,"packing_error_bound_failures":errors,"packing_max_abs_error":pack_max,"attention_max_abs_error_vs_f64":max_error,"attention_gpu_wall_ms":elapsed,"full_kv_bytes":count*1280}).serialize(&serde_wasm_bindgen::Serializer::json_compatible()).map_err(err)?)
    }

    pub fn set_kv_bits(&mut self,bits:u32)->Result<()> {
        if !matches!(bits,4|32){return Err(err("kv_bits must be 4 or 32"));}
        let full=self.capacity as u64*if bits==4 {1280}else{8192};
        if full>self.device.limits().max_storage_buffer_binding_size as u64{return Err(err("reload the engine with this KV precision"));}
        self.prefixes.clear();self.kv_bits=bits;self.pos=0;self.kv_capacity=self.capacity.min(256);self.graphs.clear();
        let size=self.kv_capacity as u64*if bits==4 {1280}else{8192};
        let fresh:Vec<_>=(0..self.layers.len()).map(|_|self.buffer("changed KV precision",size,wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST)).collect();
        for(l,kv)in self.layers.iter_mut().zip(fresh){l.kv=kv;}self.kv_bytes=size*self.layers.len() as u64;Ok(())
    }
    pub fn set_prefill_attention(&mut self, end:u32, sparse:bool, alpha:f32)->Result<()> {
        if end>self.capacity||!alpha.is_finite()||alpha<=0.0||alpha>1.0{return Err(err("prefill alpha must be (0,1] and end <= context"));}
        self.prefill_end=end;self.sparse_prefill=sparse;self.sparse_alpha=alpha;self.graphs.clear();Ok(())
    }
    pub fn set_tiled_attention(&mut self, force:bool){self.force_tiled=force;self.graphs.clear();}
    /// Pin only the validated, immutable initial system block, independently of chat KV.
    pub fn save_prefix(&mut self,slot:u32,tokens:u32)->Result<()> {
        self.check_prefix(slot,tokens)?;
        if tokens>self.pos {return Err(err("system prefix has not been computed"));}
        let group=if self.kv_bits==4 {5}else{1};let width=if self.kv_bits==4 {160}else{1024};
        let size=tokens as u64*width as u64*8;
        let layers:Vec<_>=self.layers.iter().map(|_|self.buffer("pinned system KV",size,wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST)).collect();
        let copies:Vec<_>=self.layers.iter().zip(&layers).map(|(l,b)|self.dispatch("kv_copy",&[&l.kv,b],[(tokens*width*2+255)/256,1,1],[self.kv_capacity,tokens,tokens,group])).collect();
        let mut encoder=self.device.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());for d in &copies{pass.set_pipeline(&d.handle);pass.set_bind_group(0,&d.group,&[]);pass.dispatch_workgroups(d.dims[0],1,1);}}
        self.queue.submit(Some(encoder.finish()));
        self.prefixes.insert(slot,KvPrefix{tokens,bits:self.kv_bits,layers});Ok(())
    }
    pub fn restore_prefix(&mut self,slot:u32)->Result<u32> {
        let prefix=self.prefixes.get(&slot).ok_or_else(||err("unknown system prefix"))?;
        if prefix.bits!=self.kv_bits {return Err(err("prefix KV precision changed"));}
        let tokens=prefix.tokens;self.ensure_kv_capacity(tokens)?;
        let prefix=&self.prefixes[&slot];let group=if self.kv_bits==4 {5}else{1};let width=if self.kv_bits==4 {160}else{1024};
        let copies:Vec<_>=prefix.layers.iter().zip(&self.layers).map(|(b,l)|self.dispatch("kv_copy",&[b,&l.kv],[(tokens*width*2+255)/256,1,1],[tokens,self.kv_capacity,tokens,group])).collect();
        let mut encoder=self.device.create_command_encoder(&Default::default());
        {let mut pass=encoder.begin_compute_pass(&Default::default());for d in &copies{pass.set_pipeline(&d.handle);pass.set_bind_group(0,&d.group,&[]);pass.dispatch_workgroups(d.dims[0],1,1);}}
        self.queue.submit(Some(encoder.finish()));self.pos=tokens;Ok(tokens)
    }
    pub fn drop_prefix(&mut self,slot:u32){self.prefixes.remove(&slot);}
    /// Compact layer payloads; no capacity padding, conversation tokens or logits.
    pub async fn export_prefix(&self,slot:u32)->Result<Vec<u8>> {
        let prefix=self.prefixes.get(&slot).ok_or_else(||err("unknown system prefix"))?;
        let layer_size=prefix.tokens as u64*if prefix.bits==4 {1280}else{8192};
        let size=layer_size*prefix.layers.len() as u64;
        let staging=self.buffer("system KV readback",size,wgpu::BufferUsages::COPY_DST|wgpu::BufferUsages::MAP_READ);
        let mut encoder=self.device.create_command_encoder(&Default::default());
        for(i,b)in prefix.layers.iter().enumerate(){encoder.copy_buffer_to_buffer(b,0,&staging,i as u64*layer_size,layer_size);}
        self.queue.submit(Some(encoder.finish()));
        let(tx,rx)=futures_channel::oneshot::channel();staging.slice(..).map_async(wgpu::MapMode::Read,move|r|{let _=tx.send(r);});
        rx.await.map_err(err)?.map_err(err)?;
        let bytes=staging.slice(..).get_mapped_range().to_vec();staging.unmap();Ok(bytes)
    }
    pub fn import_prefix(&mut self,slot:u32,tokens:u32,bits:u32,bytes:Vec<u8>)->Result<()> {
        self.check_prefix(slot,tokens)?;
        if bits!=self.kv_bits||bytes.len() as u64!=self.prefix_size(tokens){return Err(err("invalid system KV payload"));}
        let size=bytes.len()/self.layers.len();
        let layers=bytes.chunks_exact(size).map(|chunk|self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor{label:Some("restored system KV"),contents:chunk,usage:wgpu::BufferUsages::STORAGE|wgpu::BufferUsages::COPY_SRC|wgpu::BufferUsages::COPY_DST})).collect();
        self.prefixes.insert(slot,KvPrefix{tokens,bits,layers});Ok(())
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
            gpu_prefix_bytes:self.prefixes.values().map(|p|self.prefix_size(p.tokens)).sum(),
            gpu_scratch_bytes: self.scratch_bytes,
            wasm_linear_bytes: bytes,
            context: self.capacity,
            batch: self.batch,
            kv_capacity_tokens: self.kv_capacity,
            used_tokens: self.pos,
            kv_bits:self.kv_bits,
            gpu_kv_full_context_bytes:self.capacity as u64*28*if self.kv_bits==4 {1280}else{8192},
            yarn_factor:if self.capacity>32768 {16.0}else{4.0},
        })
        .map_err(err)
    }
    /// Returns [token-id-as-u32-bits, score] pairs from GPU partitions, globally sorted.
    pub async fn forward(&mut self, tokens: Vec<u32>, top_k: u32) -> Result<JsValue> {
        let data = self.execute(&tokens, false, top_k, false).await?;
        let mut candidates: Vec<(u32, f32)> = data
            .chunks_exact(2)
            .map(|p| (p[0].to_bits(), p[1]))
            .filter(|p| p.0 < 151669 && p.1.is_finite())
            .collect();
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| b.0.cmp(&a.0)));
        candidates.truncate(top_k.clamp(1, 64) as usize);
        serde_wasm_bindgen::to_value(&candidates).map_err(err)
    }
    /// Candidate rows for every input position, used for exact lookup-draft verification.
    pub async fn forward_all(&mut self, tokens: Vec<u32>, top_k: u32) -> Result<JsValue> {
        let top_k=top_k.clamp(1,64);
        let data=self.execute(&tokens,false,top_k,true).await?;
        let rows:Vec<Vec<(u32,f32)>>=data.chunks_exact(64*top_k as usize*2).map(|row|{
            let mut candidates:Vec<(u32,f32)>=row.chunks_exact(2).map(|p|(p[0].to_bits(),p[1])).filter(|p|p.0<151669&&p.1.is_finite()).collect();
            candidates.sort_unstable_by(|a,b|b.1.total_cmp(&a.1).then_with(||b.0.cmp(&a.0)));
            candidates.truncate(top_k as usize);candidates
        }).collect();
        serde_wasm_bindgen::to_value(&rows).map_err(err)
    }
    pub async fn logits(&mut self, token: u32) -> Result<Vec<f32>> {
        self.execute(&[token], true, 1, false).await
    }
    pub async fn logits_chunk(&mut self, tokens: Vec<u32>) -> Result<Vec<f32>> {
        self.execute(&tokens, true, 1, false).await
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
                    pass.set_pipeline(&d.handle);
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
            pass.set_pipeline(&d.handle);
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
        self.prepare_reference().await?;
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
        let graph = self.graph_mode(1,false,false);
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
                pass.set_pipeline(&d.handle);
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

#[wasm_bindgen]
pub fn ui_core(request: &str) -> std::result::Result<String, JsValue> {
    bonsai_ui_core::dispatch(request).map_err(|e| JsValue::from_str(&e))
}
