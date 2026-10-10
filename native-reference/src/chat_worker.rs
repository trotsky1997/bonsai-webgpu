//! Persistent JSONL transport for the local chat UI. Inference stays in Rust.
use super::*;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    mpsc, Arc,
};

fn emit(value: Value) -> Result<()> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, &value)?;
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}

fn cancelled(cancel: &AtomicU64, id: u64) -> bool {
    let value = cancel.load(Ordering::Relaxed);
    value == id || value == u64::MAX
}

pub(super) trait ChatModel {
    fn config(&self) -> &Config;
    fn tokenizer(&self) -> &Tokenizer;
    fn forward(&mut self, token: u32) -> Result<Vec<f32>>;
    fn truncate_cache(&mut self, prefix: usize) -> Result<()>;
    fn reset_cache(&mut self);
    fn memory_info(&self) -> Value {
        Value::Null
    }
}
impl ChatModel for Model {
    fn config(&self) -> &Config {
        &self.cfg
    }
    fn tokenizer(&self) -> &Tokenizer {
        &self.tok
    }
    fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        Model::forward(self, token)
    }
    fn reset_cache(&mut self) {
        Model::reset_cache(self);
    }
    fn truncate_cache(&mut self, prefix: usize) -> Result<()> {
        let stride = self.cfg.num_key_value_heads * self.cfg.head_dim;
        for layer in &mut self.layers {
            layer.keys.truncate(prefix * stride);
            layer.values.truncate(prefix * stride);
        }
        self.pos = prefix;
        Ok(())
    }
}

pub(super) fn run() -> Result<()> {
    let dir = PathBuf::from(
        std::env::args_os()
            .nth(2)
            .context("--chat-worker needs model directory")?,
    );
    #[cfg(feature = "mlx")]
    if std::env::var("BONSAI_BACKEND").as_deref() == Ok("mlx")
        || std::env::args()
            .next()
            .is_some_and(|s| s.contains("bonsai-mlx-chat-worker"))
    {
        let mut model = mlx_native::MlxModel::load(&dir)?;
        return serve(
            &mut model,
            json!({"backend":"mlx-metal","kernel":"signmag-g32-metal","activation_bits":4,"weight_bits":1,"threads":0}),
        );
    }
    let threads: usize = std::env::var("RAYON_NUM_THREADS")
        .unwrap_or_else(|_| "12".into())
        .parse()?;
    if threads == 0 {
        bail!("RAYON_NUM_THREADS must be positive");
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let precision = ActivationPrecision {
        bits: 4,
        group_size: 32,
        a2_scheme: A2Scheme::Escape,
        sidecar_cache: 0,
        fast_reduction: false,
        a4_kernel: a4::Kernel::SignmagExact,
        a4_row_tile: 4,
    };
    let mut model = Model::load(&dir, precision)?;
    serve(
        &mut model,
        json!({"backend":"rust-cpu","threads":threads,"kernel":"signmag-exact","row_tile":4}),
    )
}

fn serve<M: ChatModel>(model: &mut M, mut info: Value) -> Result<()> {
    let context: usize = std::env::var("BONSAI_CONTEXT")
        .unwrap_or_else(|_| "8192".into())
        .parse()?;
    if context < 2 || context > model.config().max_position_embeddings {
        bail!(
            "BONSAI_CONTEXT must be between 2 and {}",
            model.config().max_position_embeddings
        );
    }
    let (sender, receiver) = mpsc::channel();
    let cancel = Arc::new(AtomicU64::new(0));
    let input_cancel = cancel.clone();
    // Cancellation is read concurrently, including during prompt evaluation.
    std::thread::spawn(move || {
        for line in io::stdin().lock().lines() {
            let Ok(line) = line else {
                break;
            };
            match serde_json::from_str::<Value>(&line) {
                Ok(value) if value["op"] == "cancel" => {
                    if let Some(id) = value["id"].as_u64() {
                        input_cancel.store(id, Ordering::Relaxed);
                    }
                }
                Ok(value) => {
                    if sender.send(value).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = emit(json!({"type":"error","id":null,"message":e.to_string()}));
                }
            }
        }
        input_cancel.store(u64::MAX, Ordering::Relaxed);
    });
    info["type"] = json!("ready");
    info["context"] = json!(context);
    emit(info)?;
    let mut cached_ids: Vec<u32> = Vec::new();
    for request in receiver {
        let id = request["id"].as_u64().context("request id required")?;
        let result = if request["op"] == "reset" {
            model.reset_cache();
            cached_ids.clear();
            emit(json!({"type":"reset","id":id,"memory":model.memory_info()}))
        } else if request["op"] == "diagnose" {
            model.reset_cache();
            cached_ids.clear();
            diagnose(model, &request, context)
        } else {
            generate(model, &mut cached_ids, &request, &cancel, context)
        };
        if let Err(error) = result {
            model.reset_cache();
            cached_ids.clear();
            emit(json!({"type":"error","id":id,"message":error.to_string()}))?;
        }
    }
    Ok(())
}

fn diagnose<M: ChatModel>(model: &mut M, req: &Value, context: usize) -> Result<()> {
    let id = req["id"].as_u64().context("id required")?;
    let ids: Vec<u32> = if let Some(tokens) = req["tokens"].as_array() {
        tokens
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| u32::try_from(n).ok())
                    .context("invalid token")
            })
            .collect::<Result<_>>()?
    } else {
        model
            .tokenizer()
            .encode(req["text"].as_str().context("text required")?, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
            .to_vec()
    };
    if ids.len() < 2 || ids.len() > context {
        bail!("diagnostic token count out of range");
    }
    let mut nll = Vec::new();
    let mut predictions = Vec::new();
    let mut targets = Vec::new();
    for i in 0..ids.len() - 1 {
        let logits = model.forward(ids[i])?;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let sum: f64 = logits.iter().map(|x| (*x as f64 - max).exp()).sum();
        let target = *logits
            .get(ids[i + 1] as usize)
            .context("target outside vocabulary")?;
        nll.push(sum.ln() + max - target as f64);
        predictions.push(
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0,
        );
        targets.push(target);
    }
    emit(
        json!({"type":"diagnosed","id":id,"result":{"tokens":ids,"mean_nll":nll.iter().sum::<f64>()/nll.len() as f64,"nll":nll,"predictions":predictions,"target_logits":targets,"memory":model.memory_info()}}),
    )
}

fn generate<M: ChatModel>(
    model: &mut M,
    cached_ids: &mut Vec<u32>,
    req: &Value,
    cancel: &AtomicU64,
    context: usize,
) -> Result<()> {
    let id = req["id"].as_u64().context("request id required")?;
    let prompt = req["prompt"].as_str().context("prompt required")?;
    let ids = model
        .tokenizer()
        .encode(prompt, false)
        .map_err(|e| anyhow::anyhow!(e.to_string()))?
        .get_ids()
        .to_vec();
    if ids.is_empty() {
        bail!("prompt is empty");
    }
    if ids.len() >= context {
        bail!("prompt has {} tokens; context limit is {context}; start a new chat or shorten the history", ids.len());
    }
    if req["op"] == "tokenize" {
        return emit(json!({"type":"tokens","id":id,"tokens":ids}));
    }
    let max_tokens = (req["max_tokens"].as_u64().unwrap_or(512) as usize).min(context - ids.len());
    let temperature = req["temperature"].as_f64().unwrap_or(0.5) as f32;
    let top_k = (req["top_k"].as_u64().unwrap_or(20) as usize)
        .max(1)
        .min(model.config().vocab_size);
    let top_p = req["top_p"].as_f64().unwrap_or(0.85) as f32;
    let repeat_penalty = req["repeat_penalty"].as_f64().unwrap_or(1.0) as f32;
    let presence_penalty = req["presence_penalty"].as_f64().unwrap_or(0.0) as f32;
    let frequency_penalty = req["frequency_penalty"].as_f64().unwrap_or(0.0) as f32;
    let repeat_last_n = req["repeat_last_n"].as_u64().unwrap_or(64) as usize;
    // Reuse only the identical token prefix. Truncation also isolates unrelated chats.
    // Re-evaluate at least the last prompt token to obtain its logits.
    let prefix = cached_ids
        .iter()
        .zip(&ids)
        .take_while(|(a, b)| a == b)
        .count()
        .min(ids.len() - 1);
    model.truncate_cache(prefix)?;
    cached_ids.truncate(prefix);
    let prefill = Instant::now();
    emit(
        json!({"type":"progress","id":id,"prompt_tokens":ids.len(),"cached_tokens":prefix,"processed_tokens":prefix}),
    )?;
    let mut logits = Vec::new();
    for (index, token) in ids.iter().enumerate().skip(prefix) {
        if cancelled(cancel, id) {
            return emit(
                json!({"type":"done","id":id,"finish_reason":"stop","prompt_tokens":ids.len(),"completion_tokens":0,"cached_tokens":prefix,"prefill_ms":prefill.elapsed().as_secs_f64()*1000.0,"decode_ms":0.0}),
            );
        }
        logits = model.forward(*token)?;
        cached_ids.push(*token);
        if (index + 1) % 16 == 0 {
            emit(
                json!({"type":"progress","id":id,"prompt_tokens":ids.len(),"cached_tokens":prefix,"processed_tokens":index+1}),
            )?;
        }
    }
    let prefill_ms = prefill.elapsed().as_secs_f64() * 1000.0;
    emit(
        json!({"type":"prefilled","id":id,"prompt_tokens":ids.len(),"cached_tokens":prefix,"prefill_ms":prefill_ms}),
    )?;
    // Clone once to keep streaming decoder's immutable borrow independent of model.forward.
    let tokenizer = model.tokenizer().clone();
    let mut decoder = tokenizer.decode_stream(true);
    let mut history = ids;
    let start = Instant::now();
    let mut generated = 0;
    let mut generated_ids = Vec::new();
    let mut first_token_ms = None;
    let mut finish = "length";
    for step in 0..max_tokens {
        if cancelled(cancel, id) {
            finish = "stop";
            break;
        }
        if repeat_penalty != 1.0 || presence_penalty != 0.0 || frequency_penalty != 0.0 {
            let mut counts = HashMap::<u32, usize>::new();
            for token in history.iter().rev().take(repeat_last_n) {
                *counts.entry(*token).or_default() += 1;
            }
            for (token, count) in counts {
                let value = &mut logits[token as usize];
                *value = if *value > 0.0 {
                    *value / repeat_penalty
                } else {
                    *value * repeat_penalty
                };
                *value -= presence_penalty + frequency_penalty * count as f32;
            }
        }
        let token = sample(&logits, temperature, top_k, top_p);
        if req["ignore_eos"] != true
            && (token == model.config().eos_token_id
                || tokenizer.token_to_id("<|endoftext|>") == Some(token))
        {
            finish = "stop";
            break;
        }
        generated += 1;
        generated_ids.push(token);
        if first_token_ms.is_none() {
            first_token_ms = Some(prefill_ms + start.elapsed().as_secs_f64() * 1000.0);
        }
        history.push(token);
        if let Some(text) = decoder
            .step(token)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
        {
            emit(json!({"type":"delta","id":id,"text":text}))?;
        }
        if step + 1 < max_tokens {
            logits = model.forward(token)?;
            cached_ids.push(token);
        }
    }
    emit(
        json!({"type":"done","id":id,"finish_reason":finish,"prompt_tokens":history.len()-generated,"completion_tokens":generated,"cached_tokens":prefix,"prefill_ms":prefill_ms,"decode_ms":start.elapsed().as_secs_f64()*1000.0,"ttft_ms":first_token_ms,"tokens":generated_ids,"memory":model.memory_info()}),
    )
}
