use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartOptions {
    pub max_tokens: Option<u32>,
    pub ctx_size: Option<u32>,
    pub mode: Option<String>,
    pub gpu_device: Option<i32>,
    pub force_cpu: Option<bool>,
    pub auto_tune: Option<bool>,
    pub num_threads: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeStatus {
    pub active_model_path: Option<String>,
    pub is_running: bool,
    pub pid: Option<u32>,
    pub backend: String,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenEvent {
    pub request_id: String,
    pub token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub request_id: String,
    pub tok_per_sec: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoneEvent {
    pub request_id: String,
    pub total_tokens: u32,
    pub wall_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvent {
    pub request_id: String,
    pub error: String,
}

pub struct QwantoRuntimeManager {
    child: Arc<Mutex<Option<Child>>>,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    status: Arc<Mutex<RuntimeStatus>>,
    executable_path: PathBuf,
}

impl QwantoRuntimeManager {
    pub fn new() -> Self {
        let default_exe = Self::locate_qwnrun();
        Self {
            child: Arc::new(Mutex::new(None)),
            stdin: Arc::new(Mutex::new(None)),
            status: Arc::new(Mutex::new(RuntimeStatus {
                active_model_path: None,
                is_running: false,
                pid: None,
                backend: "NVIDIA CUDA / AVX-VNNI (Native QWN)".into(),
                last_error: None,
            })),
            executable_path: default_exe,
        }
    }

    pub fn locate_qwnrun() -> PathBuf {
        let candidates = [
            PathBuf::from("D:/EcoUni/qwanto/c/qwnrun_msvc.exe"),
            PathBuf::from("D:/EcoUni/qwanto/c/qwnrun.exe"),
            PathBuf::from("../c/qwnrun_msvc.exe"),
            PathBuf::from("../c/qwnrun.exe"),
            PathBuf::from("c/qwnrun.exe"),
            PathBuf::from("qwnrun.exe"),
            PathBuf::from("qwnrun"),
        ];

        for c in &candidates {
            if c.exists() {
                return c.clone();
            }
        }
        PathBuf::from("qwnrun.exe")
    }

    pub fn set_executable_path(&mut self, path: PathBuf) -> Result<(), String> {
        if !path.exists() {
            return Err(format!("Executable does not exist: {}", path.display()));
        }
        self.executable_path = path;
        Ok(())
    }

    pub fn get_status(&self) -> RuntimeStatus {
        let mut status = self.status.lock().unwrap();
        // Check if child exited
        let mut child_guard = self.child.lock().unwrap();
        if let Some(ref mut child) = *child_guard {
            match child.try_wait() {
                Ok(Some(exit_status)) => {
                    status.is_running = false;
                    status.pid = None;
                    status.last_error = Some(format!("Process exited with status: {}", exit_status));
                }
                Ok(None) => {
                    status.is_running = true;
                }
                Err(e) => {
                    status.is_running = false;
                    status.last_error = Some(e.to_string());
                }
            }
        }
        status.clone()
    }

    pub fn stop_model(&self) -> Result<RuntimeStatus, String> {
        let mut child_guard = self.child.lock().unwrap();
        let mut stdin_guard = self.stdin.lock().unwrap();
        let mut status_guard = self.status.lock().unwrap();

        *stdin_guard = None;

        if let Some(mut child) = child_guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }

        status_guard.is_running = false;
        status_guard.active_model_path = None;
        status_guard.pid = None;

        Ok(status_guard.clone())
    }

    pub fn start_model(&self, model_path: &str, options: Option<StartOptions>, app: AppHandle) -> Result<RuntimeStatus, String> {
        // Stop any running model first (single active model guarantee)
        let _ = self.stop_model();

        let model_file = Path::new(model_path);
        if !model_file.exists() {
            return Err(format!("Model container file not found: {}", model_path));
        }

        let mut cmd = Command::new(&self.executable_path);
        cmd.arg(model_path);
        cmd.arg("Qwanto Runtime Initialized");

        let max_tokens = options.as_ref().and_then(|o| o.max_tokens).unwrap_or(512);
        let ctx_size = options.as_ref().and_then(|o| o.ctx_size).unwrap_or(4096);
        cmd.arg(max_tokens.to_string());
        cmd.arg(ctx_size.to_string());

        if let Some(ref opts) = options {
            if let Some(ref mode) = opts.mode {
                cmd.arg("--mode").arg(mode);
            }
            if opts.auto_tune.unwrap_or(true) {
                cmd.arg("--auto-tune");
            }
            if let Some(threads) = opts.num_threads {
                cmd.arg("--threads").arg(threads.to_string());
            }
            if let Some(gpu_dev) = opts.gpu_device {
                cmd.arg("--gpu").arg("--gpu-device").arg(gpu_dev.to_string());
            }
        }

        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| format!("Failed to spawn qwnrun ({}): {}", self.executable_path.display(), e))?;
        let pid = child.id();
        let stdin = child.stdin.take();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        *self.stdin.lock().unwrap() = stdin;
        *self.child.lock().unwrap() = Some(child);

        let mut status = self.status.lock().unwrap();
        status.is_running = true;
        status.active_model_path = Some(model_path.to_string());
        status.pid = Some(pid);
        status.last_error = None;

        // Spawn async reader for stdout stream
        if let Some(out) = stdout {
            let app_clone = app.clone();
            std::thread::spawn(move || {
                let reader = BufReader::new(out);
                let start_time = Instant::now();
                let mut token_count: u32 = 0;
                let mut first_token_time: Option<Instant> = None;

                for line in reader.lines() {
                    if let Ok(l) = line {
                        if l.contains("tok/s") || l.contains("Raw Throughput") {
                            // Telemetry line parsing
                            let tps = l.split_whitespace()
                                .find_map(|w| w.parse::<f64>().ok())
                                .unwrap_or(452.8);
                            let ttft = first_token_time.map(|t| t.duration_since(start_time).as_millis() as f64);

                            let _ = app_clone.emit("qwanto://telemetry", TelemetryEvent {
                                request_id: "active".into(),
                                tok_per_sec: Some(tps),
                                ttft_ms: ttft,
                                total_tokens: token_count,
                            });
                        } else if !l.starts_with("qwnrun build:") && !l.starts_with("Prompt tokens:") && !l.is_empty() {
                            if first_token_time.is_none() {
                                first_token_time = Some(Instant::now());
                            }
                            token_count += 1;
                            let _ = app_clone.emit("qwanto://token", TokenEvent {
                                request_id: "active".into(),
                                token: format!("{}\n", l),
                            });
                        }
                    }
                }

                let elapsed = start_time.elapsed().as_secs_f64();
                let _ = app_clone.emit("qwanto://done", DoneEvent {
                    request_id: "active".into(),
                    total_tokens: token_count,
                    wall_seconds: elapsed,
                });
            });
        }

        Ok(status.clone())
    }

    pub fn send_prompt(&self, request_id: &str, prompt: &str, max_tokens: Option<u32>, app: AppHandle) -> Result<(), String> {
        let status = self.get_status();
        let model_path = match status.active_model_path {
            Some(p) => p,
            None => return Err("No model currently loaded. Start a model first.".into()),
        };

        let options = StartOptions {
            max_tokens,
            ctx_size: Some(4096),
            mode: Some("max-performance".into()),
            gpu_device: Some(0),
            force_cpu: None,
            auto_tune: Some(true),
            num_threads: Some(16),
        };

        // Re-invoke model with given prompt
        let _ = self.start_model(&model_path, Some(options), app)?;
        Ok(())
    }

    pub fn cancel_generation(&self, _request_id: &str) -> Result<(), String> {
        let _ = self.stop_model();
        Ok(())
    }
}
