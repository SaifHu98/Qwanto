use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue<T> {
    pub value: Option<T>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySnapshot {
    pub timestamp: String,
    pub generation_tokens: u32,
    pub tok_per_sec: MetricValue<f64>,
    pub ttft_ms: MetricValue<f64>,
    pub cpu_cores: Option<u32>,
    pub cpu_model: String,
    pub gpu_device: String,
    pub vram_used_gb: MetricValue<f64>,
    pub vram_total_gb: MetricValue<f64>,
    pub nvme_bandwidth_mb_s: MetricValue<f64>,
}

pub struct TelemetryCollector;

impl TelemetryCollector {
    pub fn get_snapshot(tok_per_sec: Option<f64>, ttft_ms: Option<f64>, tokens: u32) -> TelemetrySnapshot {
        TelemetrySnapshot {
            timestamp: chrono_or_simple_ts(),
            generation_tokens: tokens,
            tok_per_sec: MetricValue {
                value: tok_per_sec,
                status: if tok_per_sec.is_some() { "measured".into() } else { "unavailable (awaiting prompt generation)".into() },
            },
            ttft_ms: MetricValue {
                value: ttft_ms,
                status: if ttft_ms.is_some() { "measured".into() } else { "unavailable (awaiting prompt generation)".into() },
            },
            cpu_cores: std::thread::available_parallelism().map(|p| p.get() as u32).ok(),
            cpu_model: "Host CPU (Auto-Dispatched SIMD)".into(),
            gpu_device: "Auto-Selected GPU or Native CPU".into(),
            vram_used_gb: MetricValue {
                value: None,
                status: "unavailable (NVML polling inactive)".into(),
            },
            vram_total_gb: MetricValue {
                value: None,
                status: "unavailable".into(),
            },
            nvme_bandwidth_mb_s: MetricValue {
                value: None,
                status: "unavailable (disk I/O sensor inactive)".into(),
            },
        }
    }
}

fn chrono_or_simple_ts() -> String {
    format!("T+{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs())
}
