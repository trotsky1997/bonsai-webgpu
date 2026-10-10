//! Offline regression of the ternary scale after removing two dynamic outliers.
//! Inference uses linear top-two selection and four distribution features.
use super::*;
use serde::Serialize;

pub(crate) fn fingerprint(bytes: &[u8]) -> String {
    let mut value = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        value = (value ^ b as u64).wrapping_mul(0x100_0000_01b3);
    }
    format!("fnv1a64:{value:016x}")
}

pub(crate) fn features(chunk: &[f32]) -> ([usize; 2], [f32; 4], f32) {
    features_with_threshold(chunk, 0.6)
}
pub(crate) fn features_with_threshold(
    chunk: &[f32],
    threshold: f32,
) -> ([usize; 2], [f32; 4], f32) {
    let (mut first, mut second) = (-1.0f32, -1.0f32);
    let (mut i0, mut i1) = (0, 0);
    for (i, &x) in chunk.iter().enumerate() {
        let m = x.abs();
        if m > first {
            second = first;
            i1 = i0;
            first = m;
            i0 = i;
        } else if m > second {
            second = m;
            i1 = i;
        }
    }
    let (mut sum, mut square, mut max) = (0.0f64, 0.0f64, 0.0f32);
    for (i, &x) in chunk.iter().enumerate() {
        if i == i0 || i == i1 {
            continue;
        }
        let m = x.abs();
        sum += m as f64;
        square += (m as f64) * (m as f64);
        max = max.max(m);
    }
    let mean = sum / (chunk.len() - 2) as f64;
    let rms = (square / (chunk.len() - 2) as f64).sqrt();
    let mut tail_sum = 0.0f64;
    let mut tail_count = 0;
    for (i, &x) in chunk.iter().enumerate() {
        if i != i0 && i != i1 && x.abs() as f64 >= threshold as f64 * rms {
            tail_sum += x.abs() as f64;
            tail_count += 1;
        }
    }
    (
        [i0, i1],
        [
            mean as f32,
            rms as f32,
            max,
            if tail_count > 0 {
                (tail_sum / tail_count as f64) as f32
            } else {
                0.0
            },
        ],
        first,
    )
}

#[derive(Serialize, Deserialize)]
pub(crate) struct Site {
    pub name: String,
    pub shared: [f32; 4],
    pub shared_threshold: f32,
    pub thresholds: Vec<f32>,
    pub coefficients: Vec<[f32; 4]>,
    pub observations_per_group: usize,
}
#[derive(Serialize, Deserialize)]
pub(crate) struct Calibration {
    pub schema: String,
    pub model_fingerprint: String,
    pub corpus_fingerprint: String,
    pub activation_group_size: usize,
    pub escapes_per_group: usize,
    pub feature_names: Vec<String>,
    pub alpha_factor: f32,
    pub per_group: bool,
    pub rule: String,
    pub tokens: usize,
    pub sites: Vec<Site>,
}
impl Calibration {
    pub fn validate(&self, model: &Model) -> Result<()> {
        if self.schema != "bonsai-sidecar-calibration-v1"
            || self.model_fingerprint != model.fingerprint
        {
            bail!("calibration schema or model fingerprint does not match");
        }
        let shapes = site_shapes(&model.cfg);
        if self.activation_group_size != 32
            || self.escapes_per_group != 2
            || self
                .feature_names
                .last()
                .is_none_or(|s| s != "mean_abs_above_0.6_rms_core")
            || self.sites.len() != shapes.len()
            || !matches!(
                self.rule.as_str(),
                "regression" | "threshold" | "threshold-refine1" | "threshold-refine2"
            )
            || !self.alpha_factor.is_finite()
            || self.alpha_factor <= 0.0
        {
            bail!("invalid calibration dimensions or alpha factor");
        }
        for (site, (name, groups)) in self.sites.iter().zip(shapes) {
            if site.name != name
                || site.coefficients.len() != groups
                || site.thresholds.len() != groups
                || !site
                    .thresholds
                    .iter()
                    .chain(std::iter::once(&site.shared_threshold))
                    .all(|t| t.is_finite() && *t > 0.0)
                || !site
                    .shared
                    .iter()
                    .chain(site.coefficients.iter().flatten())
                    .all(|x| x.is_finite())
            {
                bail!("invalid calibration site {}", name);
            }
        }
        Ok(())
    }
    pub fn coefficients(&self, site: usize, group: usize) -> [f32; 4] {
        if self.per_group {
            self.sites[site].coefficients[group]
        } else {
            self.sites[site].shared
        }
    }
}

#[derive(Clone, Default)]
struct Stats {
    xx: [[f64; 4]; 4],
    xy: [f64; 4],
    count: usize,
    ratio_total: f64,
}
impl Stats {
    fn add(&mut self, x: [f32; 4], y: f32) {
        for i in 0..4 {
            self.xy[i] += x[i] as f64 * y as f64;
            for j in 0..4 {
                self.xx[i][j] += x[i] as f64 * x[j] as f64;
            }
        }
        self.count += 1;
        if x[1] > 0.0 {
            self.ratio_total += 0.5 * y as f64 / x[1] as f64;
        } else {
            self.ratio_total += 0.6;
        }
    }
    fn merge(&mut self, other: &Self) {
        for i in 0..4 {
            self.xy[i] += other.xy[i];
            for j in 0..4 {
                self.xx[i][j] += other.xx[i][j];
            }
        }
        self.count += other.count;
        self.ratio_total += other.ratio_total;
    }
    fn solve(&self, prior: [f32; 4], ridge: f64) -> [f32; 4] {
        let lambda = (0..4).map(|i| self.xx[i][i]).sum::<f64>() * ridge / 4.0 + 1e-20;
        let mut a = [[0.0f64; 5]; 4];
        for i in 0..4 {
            for j in 0..4 {
                a[i][j] = self.xx[i][j];
            }
            a[i][i] += lambda;
            a[i][4] = self.xy[i] + lambda * prior[i] as f64;
        }
        for i in 0..4 {
            let pivot = (i..4)
                .max_by(|&j, &k| a[j][i].abs().total_cmp(&a[k][i].abs()))
                .unwrap();
            a.swap(i, pivot);
            let divisor = a[i][i];
            if divisor.abs() < 1e-30 {
                return prior;
            }
            for j in i..5 {
                a[i][j] /= divisor;
            }
            for k in 0..4 {
                if k != i {
                    let factor = a[k][i];
                    for j in i..5 {
                        a[k][j] -= factor * a[i][j];
                    }
                }
            }
        }
        std::array::from_fn(|i| a[i][4] as f32)
    }
}
pub(crate) fn site_shapes(cfg: &Config) -> Vec<(String, usize)> {
    let mut shapes = Vec::new();
    for i in 0..cfg.num_hidden_layers {
        for (name, len) in [
            ("qkv", cfg.hidden_size),
            ("o", cfg.hidden_size),
            ("gate_up", cfg.hidden_size),
            ("down", cfg.intermediate_size),
        ] {
            shapes.push((format!("layer.{i}.{name}"), len / 32));
        }
    }
    shapes.push(("lm_head".into(), cfg.hidden_size / 32));
    shapes
}
pub(crate) struct Collector {
    stats: Vec<Vec<Stats>>,
    names: Vec<String>,
    pub tokens: usize,
}
impl Collector {
    pub fn new(cfg: &Config) -> Self {
        let shapes = site_shapes(cfg);
        Self {
            stats: shapes
                .iter()
                .map(|(_, n)| vec![Stats::default(); *n])
                .collect(),
            names: shapes.into_iter().map(|(s, _)| s).collect(),
            tokens: 0,
        }
    }
    pub fn observe(&mut self, site: usize, values: &[f32], packed: &PackedActivation) {
        for (g, chunk) in values.chunks_exact(32).enumerate() {
            self.stats[site][g].add(features(chunk).1, packed.scales[g]);
        }
    }
    pub fn finish(self, model_fingerprint: String, corpus_fingerprint: String) -> Calibration {
        let sites = self
            .stats
            .into_iter()
            .zip(self.names)
            .map(|(groups, name)| {
                let mut pooled = Stats::default();
                for group in &groups {
                    pooled.merge(group);
                }
                let shared = pooled.solve([0.0, 1.2, 0.0, 0.0], 1e-5);
                let count = groups[0].count;
                Site {
                    name,
                    shared,
                    shared_threshold: (pooled.ratio_total / pooled.count as f64) as f32,
                    thresholds: groups
                        .iter()
                        .map(|s| (s.ratio_total / s.count as f64) as f32)
                        .collect(),
                    coefficients: groups.iter().map(|s| s.solve(shared, 1e-3)).collect(),
                    observations_per_group: count,
                }
            })
            .collect();
        Calibration {
            schema: "bonsai-sidecar-calibration-v1".into(),
            model_fingerprint,
            corpus_fingerprint,
            activation_group_size: 32,
            escapes_per_group: 2,
            feature_names: [
                "mean_abs_core",
                "rms_core",
                "max_abs_core",
                "mean_abs_above_0.6_rms_core",
            ]
            .map(String::from)
            .to_vec(),
            alpha_factor: 1.0,
            per_group: true,
            rule: "regression".into(),
            tokens: self.tokens,
            sites,
        }
    }
}

pub(crate) fn activation<'a>(
    values: &'a [f32],
    precision: ActivationPrecision,
    site: usize,
    cal: Option<&Calibration>,
    collector: Option<&mut Collector>,
) -> Activation<'a> {
    let mut result = if let Some(cal) = cal {
        Activation::Quantized(crate::sidecar::quantize_calibrated(values, site, cal))
    } else {
        Activation::new(values, precision)
    };
    if let Activation::Quantized(packed) = &mut result {
        packed.fast_reduction = precision.fast_reduction;
        if let Some(collector) = collector {
            collector.observe(site, values, packed);
        }
        if precision.sidecar_cache > 0 && matches!(precision.a2_scheme, A2Scheme::Sidecar(2)) {
            crate::sidecar::prepare_cache(packed, precision.sidecar_cache);
        }
    }
    result
}

pub(crate) fn collect(model: &mut Model, input: &Path, output: &Path) -> Result<()> {
    let source = std::fs::read(input)?;
    model.collector = Some(Collector::new(&model.cfg));
    for line in io::BufReader::new(source.as_slice()).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let case: DiagnosticText = serde_json::from_str(&line)?;
        let ids = model
            .tok
            .encode(case.text.as_str(), false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
            .to_vec();
        model.reset_cache();
        for &id in &ids {
            let _ = model.forward(id)?;
            model.collector.as_mut().unwrap().tokens += 1;
        }
        eprintln!("CALIBRATION name={} tokens={}", case.name, ids.len());
    }
    let collector = model.collector.take().unwrap();
    if collector.tokens == 0 {
        bail!("empty calibration corpus");
    }
    let calibration = collector.finish(model.fingerprint.clone(), fingerprint(&source));
    std::fs::write(output, serde_json::to_vec_pretty(&calibration)?)?;
    println!(
        "calibrated {} tokens, {} sites -> {}",
        calibration.tokens,
        calibration.sites.len(),
        output.display()
    );
    Ok(())
}
