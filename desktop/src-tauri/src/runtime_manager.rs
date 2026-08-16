use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};

type RequestTracking = Arc<Mutex<HashMap<String, (Instant, Option<Instant>)>>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartOptions {
    pub max_tokens: Option<u32>,
    pub ctx_size: Option<u32>,
    pub mode: Option<String>,
    pub gpu_device: Option<i32>,
    pub force_cpu: Option<bool>,
    pub auto_tune: Option<bool>,
    pub num_threads: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
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

#[derive(Debug, Clone, PartialEq)]
pub enum ProtocolFrame {
    Ready,
    Token {
        request_id: String,
        token: String,
    },
    Done {
        request_id: String,
        generated_tokens: u32,
        tok_per_sec: f64,
        prompt_tokens: u32,
        is_truncated: bool,
    },
    Error {
        request_id: String,
        reason: String,
    },
    Pong,
    Config {
        dim: u32,
        vocab: u32,
        layers: u32,
    },
    Unknown(String),
}

pub fn parse_next_protocol_frame<R: BufRead>(reader: &mut R) -> std::io::Result<Option<ProtocolFrame>> {
    let mut line = String::new();
    let n = reader.read_line(&mut line)?;
    if n == 0 {
        return Ok(None);
    }

    let trimmed = line.trim_end_matches(['\r', '\n']);
    if trimmed == "\x01\x01READY\x01\x01" || trimmed == "READY" {
        return Ok(Some(ProtocolFrame::Ready));
    }

    if trimmed == "PONG" {
        return Ok(Some(ProtocolFrame::Pong));
    }

    if trimmed.starts_with("DATA ") {
        let parts: Vec<&str> = trimmed.splitn(3, ' ').collect();
        if parts.len() == 3 {
            let request_id = parts[1].to_string();
            if let Ok(byte_count) = parts[2].parse::<usize>() {
                let mut buf = vec![0u8; byte_count];
                reader.read_exact(&mut buf)?;
                // Consume trailing newline emitted by emit_mux (putchar('\n'))
                let mut trailing = [0u8; 1];
                let _ = reader.read_exact(&mut trailing);

                let token = String::from_utf8_lossy(&buf).to_string();
                return Ok(Some(ProtocolFrame::Token { request_id, token }));
            }
        }
    }

    if trimmed.starts_with("DONE ") {
        // Format: DONE <id> STAT <generated> <tps> 0 0 <prompt_tokens> <is_truncated>
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() >= 8 && parts[2] == "STAT" {
            let request_id = parts[1].to_string();
            let generated_tokens = parts[3].parse::<u32>().unwrap_or(0);
            let tok_per_sec = parts[4].parse::<f64>().unwrap_or(0.0);
            let prompt_tokens = parts[7].parse::<u32>().unwrap_or(0);
            let is_truncated = parts.get(8).map(|&s| s == "1").unwrap_or(false);

            return Ok(Some(ProtocolFrame::Done {
                request_id,
                generated_tokens,
                tok_per_sec,
                prompt_tokens,
                is_truncated,
            }));
        }
    }

    if trimmed.starts_with("ERROR ") {
        let parts: Vec<&str> = trimmed.splitn(3, ' ').collect();
        if parts.len() >= 3 {
            return Ok(Some(ProtocolFrame::Error {
                request_id: parts[1].to_string(),
                reason: parts[2].to_string(),
            }));
        } else if parts.len() == 2 {
            return Ok(Some(ProtocolFrame::Error {
                request_id: parts[1].to_string(),
                reason: "unspecified error".to_string(),
            }));
        }
    }

    if trimmed.starts_with("CONFIG ") {
        let mut dim = 0;
        let mut vocab = 0;
        let mut layers = 0;
        for token in trimmed.split_whitespace().skip(1) {
            if let Some((k, v)) = token.split_once('=') {
                match k {
                    "dim" => dim = v.parse().unwrap_or(0),
                    "vocab" => vocab = v.parse().unwrap_or(0),
                    "layers" => layers = v.parse().unwrap_or(0),
                    _ => {}
                }
            }
        }
        return Ok(Some(ProtocolFrame::Config { dim, vocab, layers }));
    }

    Ok(Some(ProtocolFrame::Unknown(trimmed.to_string())))
}

pub struct QwantoRuntimeManager {
    child: Arc<Mutex<Option<Child>>>,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    status: Arc<Mutex<RuntimeStatus>>,
    request_tracking: RequestTracking,
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
                backend: "Unknown".into(),
                last_error: None,
            })),
            request_tracking: Arc::new(Mutex::new(HashMap::new())),
            executable_path: default_exe,
        }
    }

    pub fn locate_qwnrun() -> PathBuf {
        // 1. Packaged Mode: Check resource directory relative to current executable
        if let Ok(current_exe) = std::env::current_exe() {
            if let Some(exe_dir) = current_exe.parent() {
                let packaged_candidates = [
                    exe_dir.join("qwnrun.exe"),
                    exe_dir.join("qwnrun"),
                    exe_dir.join("resources").join("qwnrun.exe"),
                    exe_dir.join("resources").join("qwnrun"),
                ];
                for p in &packaged_candidates {
                    if p.exists() {
                        return p.clone();
                    }
                }
            }
        }

        // 2. Development Mode: Check relative workspace directories
        let dev_candidates = [
            PathBuf::from("../c/qwnrun_msvc.exe"),
            PathBuf::from("../c/qwnrun.exe"),
            PathBuf::from("../c/qwnrun"),
            PathBuf::from("c/qwnrun_msvc.exe"),
            PathBuf::from("c/qwnrun.exe"),
            PathBuf::from("c/qwnrun"),
            PathBuf::from("qwnrun.exe"),
            PathBuf::from("qwnrun"),
        ];

        for c in &dev_candidates {
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
        let mut child_guard = self.child.lock().unwrap();
        if let Some(ref mut child) = *child_guard {
            match child.try_wait() {
                Ok(Some(exit_status)) => {
                    status.is_running = false;
                    status.pid = None;
                    status.last_error = Some(format!("Process exited: {}", exit_status));
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
        status_guard.backend = "Unknown".into();

        Ok(status_guard.clone())
    }

    pub fn start_model(&self, model_path: &str, options: Option<StartOptions>, app: AppHandle) -> Result<RuntimeStatus, String> {
        // Stop any running instance cleanly
        let _ = self.stop_model();

        let model_file = Path::new(model_path);
        if !model_file.exists() {
            return Err(format!("Model container file not found: {}", model_path));
        }

        let canonical_model = model_file.canonicalize().map_err(|e| format!("Failed to canonicalize model path: {}", e))?;
        let canonical_model_str = canonical_model.to_string_lossy().to_string();

        let mut cmd = Command::new(&self.executable_path);
        cmd.arg(&canonical_model_str);
        cmd.arg("--serve");

        let ctx_size = options.as_ref().and_then(|o| o.ctx_size).unwrap_or(4096);
        let max_tokens = options.as_ref().and_then(|o| o.max_tokens).unwrap_or(512);

        cmd.env("SERVE", "1");
        cmd.env("CTX", ctx_size.to_string());
        cmd.env("NGEN", max_tokens.to_string());

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
        status.active_model_path = Some(canonical_model_str);
        status.pid = Some(pid);
        status.backend = "Unknown".into();
        status.last_error = None;

        // Spawn background parser for stdout protocol stream
        if let Some(out) = stdout {
            let app_clone = app.clone();
            let tracking_clone = Arc::clone(&self.request_tracking);
            let status_clone = Arc::clone(&self.status);

            std::thread::spawn(move || {
                let mut reader = BufReader::new(out);
                loop {
                    match parse_next_protocol_frame(&mut reader) {
                        Ok(Some(frame)) => match frame {
                            ProtocolFrame::Ready => {
                                let mut st = status_clone.lock().unwrap();
                                st.is_running = true;
                            }
                            ProtocolFrame::Token { request_id, token } => {
                                let mut track = tracking_clone.lock().unwrap();
                                if let Some((_start, first)) = track.get_mut(&request_id) {
                                    if first.is_none() {
                                        *first = Some(Instant::now());
                                    }
                                }

                                let _ = app_clone.emit("qwanto://token", TokenEvent {
                                    request_id,
                                    token,
                                });
                            }
                            ProtocolFrame::Done {
                                request_id,
                                generated_tokens,
                                tok_per_sec,
                                prompt_tokens: _,
                                is_truncated: _,
                            } => {
                                let (ttft_ms, wall_seconds) = {
                                    let mut track = tracking_clone.lock().unwrap();
                                    if let Some((start, first)) = track.remove(&request_id) {
                                        let ttft = first.map(|f| f.duration_since(start).as_secs_f64() * 1000.0);
                                        let wall = start.elapsed().as_secs_f64();
                                        (ttft, wall)
                                    } else {
                                        (None, 0.0)
                                    }
                                };

                                let _ = app_clone.emit("qwanto://telemetry", TelemetryEvent {
                                    request_id: request_id.clone(),
                                    tok_per_sec: if tok_per_sec > 0.0 { Some(tok_per_sec) } else { None },
                                    ttft_ms,
                                    total_tokens: generated_tokens,
                                });

                                let _ = app_clone.emit("qwanto://done", DoneEvent {
                                    request_id,
                                    total_tokens: generated_tokens,
                                    wall_seconds,
                                });
                            }
                            ProtocolFrame::Error { request_id, reason } => {
                                let _ = app_clone.emit("qwanto://error", ErrorEvent {
                                    request_id,
                                    error: reason,
                                });
                            }
                            ProtocolFrame::Pong | ProtocolFrame::Config { .. } | ProtocolFrame::Unknown(_) => {}
                        },
                        Ok(None) => {
                            // EOF
                            let mut st = status_clone.lock().unwrap();
                            st.is_running = false;
                            st.pid = None;
                            break;
                        }
                        Err(e) => {
                            let mut st = status_clone.lock().unwrap();
                            st.last_error = Some(format!("Protocol read error: {}", e));
                            break;
                        }
                    }
                }
            });
        }

        // Spawn background diagnostics logger for stderr
        if let Some(err) = stderr {
            let status_clone = Arc::clone(&self.status);
            std::thread::spawn(move || {
                let reader = BufReader::new(err);
                for line in reader.lines().map_while(Result::ok) {
                    if line.contains("backend=") {
                        let mut st = status_clone.lock().unwrap();
                        if line.contains("backend=CUDA") {
                            st.backend = "NVIDIA CUDA (SM89 BitDecoding)".into();
                        } else if line.contains("backend=CPU") {
                            st.backend = "Host CPU (AVX-VNNI SIMD)".into();
                        } else {
                            st.backend = line.clone();
                        }
                    }
                }
            });
        }

        Ok(status.clone())
    }

    pub fn send_prompt(
        &self,
        request_id: &str,
        prompt: &str,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
    ) -> Result<(), String> {
        let mut stdin_guard = self.stdin.lock().unwrap();
        let stdin = match *stdin_guard {
            Some(ref mut s) => s,
            None => return Err("Runtime is not running. Start a model first.".into()),
        };

        let temp = temperature.unwrap_or(0.0).clamp(0.0, 2.0);
        let topp = top_p.unwrap_or(1.0).clamp(0.0, 1.0);
        let max_tok = max_tokens.unwrap_or(512).clamp(1, 32768);

        let prompt_bytes = prompt.as_bytes();
        if prompt_bytes.len() > (16 << 20) {
            return Err("Prompt exceeds 16MB safety limit.".into());
        }

        // Protocol command: SUBMIT <id> <slot> <bytes> <max_tokens> <temp> <top_p>\n
        let submit_header = format!(
            "SUBMIT {} 0 {} {} {:.6} {:.6}\n",
            request_id,
            prompt_bytes.len(),
            max_tok,
            temp,
            topp
        );

        // Record start time for latency calculation
        {
            let mut track = self.request_tracking.lock().unwrap();
            track.insert(request_id.to_string(), (Instant::now(), None));
        }

        stdin.write_all(submit_header.as_bytes()).map_err(|e| format!("Failed to write request header: {}", e))?;
        stdin.write_all(prompt_bytes).map_err(|e| format!("Failed to write prompt payload: {}", e))?;
        stdin.write_all(b"\n").map_err(|e| format!("Failed to write request terminator: {}", e))?;
        stdin.flush().map_err(|e| format!("Failed to flush stdin: {}", e))?;

        Ok(())
    }

    pub fn cancel_generation(&self, request_id: &str) -> Result<(), String> {
        let mut stdin_guard = self.stdin.lock().unwrap();
        if let Some(ref mut stdin) = *stdin_guard {
            let cancel_cmd = format!("CANCEL {}\n", request_id);
            let _ = stdin.write_all(cancel_cmd.as_bytes());
            let _ = stdin.flush();
        }
        Ok(())
    }
}

impl Default for QwantoRuntimeManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_parse_token_frame_with_newlines() {
        let payload = "DATA req-42 12\nhello\nworld!\n";
        let mut cursor = Cursor::new(payload);
        let frame = parse_next_protocol_frame(&mut cursor).unwrap();

        assert_eq!(
            frame,
            Some(ProtocolFrame::Token {
                request_id: "req-42".into(),
                token: "hello\nworld!".into()
            })
        );
    }

    #[test]
    fn test_parse_done_frame() {
        let payload = "DONE req-42 STAT 64 45.250 0 0 16 0\n";
        let mut cursor = Cursor::new(payload);
        let frame = parse_next_protocol_frame(&mut cursor).unwrap();

        assert_eq!(
            frame,
            Some(ProtocolFrame::Done {
                request_id: "req-42".into(),
                generated_tokens: 64,
                tok_per_sec: 45.250,
                prompt_tokens: 16,
                is_truncated: false
            })
        );
    }

    #[test]
    fn test_parse_error_frame() {
        let payload = "ERROR req-42 invalid-prompt-size\n";
        let mut cursor = Cursor::new(payload);
        let frame = parse_next_protocol_frame(&mut cursor).unwrap();

        assert_eq!(
            frame,
            Some(ProtocolFrame::Error {
                request_id: "req-42".into(),
                reason: "invalid-prompt-size".into()
            })
        );
    }

    #[test]
    fn test_parse_ready_frame() {
        let payload = "\x01\x01READY\x01\x01\n";
        let mut cursor = Cursor::new(payload);
        let frame = parse_next_protocol_frame(&mut cursor).unwrap();

        assert_eq!(frame, Some(ProtocolFrame::Ready));
    }
}
