use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, mpsc};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

const READY_PREFIX: &str = "QWANTO_GATEWAY_READY ";

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[cfg(windows)]
fn configure_hidden(command: &mut Command) {
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn configure_hidden(_command: &mut Command) {}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GatewayReady {
    pub gateway: String,
    pub api_version: String,
    pub gateway_version: String,
    pub host: String,
    pub port: u16,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GatewayStatus {
    pub state: String,
    pub api_url: Option<String>,
    pub port: Option<u16>,
    pub error: Option<String>,
    pub sidecar_packaged: bool,
}

pub fn parse_ready_line(line: &str) -> Result<GatewayReady, String> {
    let payload = line
        .trim()
        .strip_prefix(READY_PREFIX)
        .ok_or_else(|| "Gateway did not publish a structured readiness line.".to_string())?;
    let ready: GatewayReady = serde_json::from_str(payload)
        .map_err(|error| format!("Invalid gateway readiness payload: {error}"))?;
    if ready.gateway != "qwanto" || ready.api_version != "1" || ready.port == 0 {
        return Err("Gateway readiness payload failed the Qwanto contract.".into());
    }
    if ready.host != "127.0.0.1" {
        return Err("Gateway sidecar must bind to 127.0.0.1.".into());
    }
    Ok(ready)
}

pub struct GatewayManager {
    child: Option<Child>,
    status: GatewayStatus,
    resource_dir: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    desktop_search_token: Option<String>,
    stderr_lines: Arc<std::sync::Mutex<VecDeque<String>>>,
}

impl GatewayManager {
    pub fn new() -> Self {
        Self {
            child: None,
            status: GatewayStatus {
                state: "stopped".into(),
                api_url: None,
                port: None,
                error: None,
                sidecar_packaged: false,
            },
            resource_dir: None,
            data_dir: None,
            desktop_search_token: None,
            stderr_lines: Arc::new(std::sync::Mutex::new(VecDeque::new())),
        }
    }

    fn find_resource(resource_dir: &Path, name: &str) -> Option<PathBuf> {
        let candidates = [
            resource_dir.join(format!("{name}.exe")),
            resource_dir.join(name),
            resource_dir.join("resources").join(format!("{name}.exe")),
            resource_dir.join("resources").join(name),
        ];
        candidates.into_iter().find(|path| path.is_file())
    }

    fn find_dev_gateway() -> Option<PathBuf> {
        let candidates = [
            PathBuf::from("c/qwanto-gateway.exe"),
            PathBuf::from("c/qwanto-gateway"),
            PathBuf::from("../c/qwanto-gateway.exe"),
            PathBuf::from("../c/qwanto-gateway"),
        ];
        candidates.into_iter().find(|path| path.is_file())
    }

    fn find_dev_script() -> Option<PathBuf> {
        let candidates = [PathBuf::from("c/openai_server.py"), PathBuf::from("../c/openai_server.py")];
        candidates.into_iter().find(|path| path.is_file())
    }

    fn find_qwnrun(resource_dir: &Path) -> Option<PathBuf> {
        let candidates = [
            resource_dir.join("qwnrun.exe"),
            resource_dir.join("qwnrun"),
            resource_dir.join("resources").join("qwnrun.exe"),
            resource_dir.join("resources").join("qwnrun"),
            PathBuf::from("c/qwnrun.exe"),
            PathBuf::from("c/qwnrun"),
            PathBuf::from("../c/qwnrun.exe"),
            PathBuf::from("../c/qwnrun"),
        ];
        candidates.into_iter().find(|path| path.is_file())
    }

    pub fn start(&mut self, resource_dir: &Path, data_dir: &Path) -> Result<GatewayStatus, String> {
        self.stop();
        self.resource_dir = Some(resource_dir.to_path_buf());
        self.data_dir = Some(data_dir.to_path_buf());
        let search_token = format!(
            "qwanto-search-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
        );
        self.desktop_search_token = Some(search_token.clone());
        if let Ok(mut lines) = self.stderr_lines.lock() {
            lines.clear();
        }
        self.status = GatewayStatus {
            state: "starting".into(),
            api_url: None,
            port: None,
            error: None,
            sidecar_packaged: false,
        };

        let model_root = data_dir.join("models");
        fs::create_dir_all(&model_root).map_err(|error| self.fail(format!("Cannot create model directory: {error}")))?;
        let ready_file = data_dir.join("gateway.ready.json");
        let qwnrun = Self::find_qwnrun(resource_dir);
        let packaged = Self::find_resource(resource_dir, "qwanto-gateway");
        let dev_binary = Self::find_dev_gateway();
        let dev_script = Self::find_dev_script();

        let mut command = if let Some(binary) = packaged.as_ref().or(dev_binary.as_ref()) {
            self.status.sidecar_packaged = packaged.is_some();
            Command::new(binary)
        } else if let Some(script) = dev_script {
            let python = if cfg!(windows) { "python" } else { "python3" };
            let mut cmd = Command::new(python);
            cmd.arg(script);
            cmd
        } else {
            return Err(self.fail("Packaged Qwanto gateway sidecar was not found.".into()));
        };

        command
            .args(["--host", "127.0.0.1", "--port", "0"])
            .arg("--ready-file")
            .arg(&ready_file)
            .env("QWANTO_DISABLE_SETTINGS", "1")
            .env("QWANTO_DESKTOP_SIDECAR", "1")
            .env("QWANTO_DESKTOP_SEARCH_TOKEN", &search_token)
            .env("QWANTO_MODEL_ROOT", &model_root)
            .env("QWANTO_MODEL_PATHS", &model_root);
        if let Some(qwnrun) = qwnrun {
            command
                .arg("--engine")
                .arg(&qwnrun)
                .env("QWANTO_QWNRUN", &qwnrun);
        }
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
        configure_hidden(&mut command);

        let mut child = command.spawn().map_err(|error| self.fail(format!("Failed to start gateway sidecar: {error}")))?;
        let stdout = child.stdout.take().ok_or_else(|| self.fail("Gateway stdout was not piped.".into()))?;
        if let Some(stderr) = child.stderr.take() {
            let stderr_lines = Arc::clone(&self.stderr_lines);
            std::thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().map_while(Result::ok) {
                    if let Ok(mut lines) = stderr_lines.lock() {
                        lines.push_back(line);
                        while lines.len() > 80 { lines.pop_front(); }
                    }
                }
            });
        }

        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            let result = reader.read_line(&mut line).map(|_| line);
            let _ = sender.send(result);
        });

        let line = match receiver.recv_timeout(Duration::from_secs(15)) {
            Ok(Ok(line)) => line,
            Ok(Err(error)) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(self.fail(format!("Could not read gateway readiness: {error}")));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(self.fail("Gateway sidecar did not become ready within 15 seconds.".into()));
            }
        };
        let ready = match parse_ready_line(&line) {
            Ok(ready) => ready,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(self.fail(error));
            }
        };

        self.child = Some(child);
        self.status = GatewayStatus {
            state: "ready".into(),
            api_url: Some(ready.url),
            port: Some(ready.port),
            error: None,
            sidecar_packaged: packaged.is_some(),
        };
        Ok(self.status.clone())
    }

    fn fail(&mut self, error: String) -> String {
        self.status.state = "failed".into();
        self.status.error = Some(error.clone());
        error
    }

    pub fn status(&mut self) -> GatewayStatus {
        let exit = self
            .child
            .as_mut()
            .and_then(|child| child.try_wait().ok().flatten());
        if let Some(exit) = exit {
            self.status.state = "failed".into();
            let diagnostics = self.stderr_lines.lock().ok().map(|lines| lines.iter().rev().take(8).cloned().collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n")).unwrap_or_default();
            self.status.error = Some(if diagnostics.is_empty() { format!("Gateway exited with {exit}") } else { format!("Gateway exited with {exit}\n{diagnostics}") });
            self.child = None;
        }
        self.status.clone()
    }

    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            #[cfg(windows)]
            {
                let pid = child.id().to_string();
                let mut taskkill = Command::new("taskkill");
                taskkill.args(["/PID", &pid, "/T", "/F"]);
                configure_hidden(&mut taskkill);
                let _ = taskkill.status();
            }
            #[cfg(not(windows))]
            {
                let _ = child.kill();
            }
            let _ = child.wait();
        }
        self.status.state = "stopped".into();
        self.status.api_url = None;
        self.status.port = None;
    }

    pub fn restart(&mut self) -> Result<GatewayStatus, String> {
        let resource_dir = self
            .resource_dir
            .clone()
            .ok_or_else(|| "Gateway has not been initialized.".to_string())?;
        let data_dir = self
            .data_dir
            .clone()
            .ok_or_else(|| "Gateway data directory is unavailable.".to_string())?;
        self.start(&resource_dir, &data_dir)
    }

    pub fn desktop_search_token(&self) -> Option<&str> {
        self.desktop_search_token.as_deref()
    }
}

impl Default for GatewayManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for GatewayManager {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_validates_ready_handshake() {
        let ready = parse_ready_line(
            r#"QWANTO_GATEWAY_READY {"gateway":"qwanto","api_version":"1","gateway_version":"0.1.0-beta.3","host":"127.0.0.1","port":43210,"url":"http://127.0.0.1:43210"}"#,
        )
        .unwrap();
        assert_eq!(ready.port, 43210);
    }

    #[test]
    fn rejects_non_loopback_ready_handshake() {
        let result = parse_ready_line(
            r#"QWANTO_GATEWAY_READY {"gateway":"qwanto","api_version":"1","gateway_version":"0.1.0-beta.3","host":"0.0.0.0","port":43210,"url":"http://0.0.0.0:43210"}"#,
        );
        assert!(result.is_err());
    }

    #[cfg(windows)]
    #[test]
    fn packaged_gateway_smoke_is_hidden_and_serves_health() {
        use std::io::{Read, Write};

        let resource_dir = match std::env::var_os("QWANTO_GATEWAY_RESOURCE_DIR") {
            Some(path) => PathBuf::from(path),
            None if std::env::var("QWANTO_REQUIRE_SIDECAR_SMOKE").ok().as_deref() == Some("1") => {
                panic!("QWANTO_GATEWAY_RESOURCE_DIR is required for the packaged gateway smoke test")
            }
            None => return,
        };
        if !resource_dir.join("qwanto-gateway.exe").is_file() {
            if std::env::var("QWANTO_REQUIRE_SIDECAR_SMOKE").ok().as_deref() == Some("1") {
                panic!("packaged gateway was not found in {}", resource_dir.display());
            }
            return;
        }
        let data_dir = std::env::temp_dir().join(format!("qwanto-gateway-smoke-{}", std::process::id()));
        let _ = fs::remove_dir_all(&data_dir);
        let _ = fs::create_dir_all(&data_dir);
        let before = forbidden_console_process_count();
        let mut manager = GatewayManager::new();
        let status = manager.start(&resource_dir, &data_dir).expect("gateway sidecar should start");
        assert_eq!(forbidden_console_process_count(), before, "desktop must not launch a console host");
        let url = status.api_url.expect("gateway URL");
        let authority = url.strip_prefix("http://").expect("loopback HTTP URL");
        let mut address = authority.split(':');
        let host = address.next().unwrap_or("127.0.0.1");
        let port = address.next().expect("gateway port").parse::<u16>().expect("valid port");
        let mut stream = std::net::TcpStream::connect((host, port)).expect("health connection");
        stream.write_all(format!("GET /health HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n").as_bytes()).unwrap();
        let mut response = String::new();
        stream.read_to_string(&mut response).unwrap();
        assert!(response.starts_with("HTTP/1.1 200"), "health response: {response}");
        manager.stop();
        let _ = fs::remove_dir_all(data_dir);
    }

    #[cfg(windows)]
    fn forbidden_console_process_count() -> usize {
        ["cmd.exe", "powershell.exe", "windowsterminal.exe"].iter().map(|name| {
            let mut command = Command::new("tasklist");
            command.arg("/FI").arg(format!("IMAGENAME eq {name}")).args(["/FO", "CSV", "/NH"]);
            configure_hidden(&mut command);
            command.output().ok().map(|output| String::from_utf8_lossy(&output.stdout).lines().filter(|line| !line.trim().is_empty() && !line.contains("INFO:")).count()).unwrap_or(0)
        }).sum()
    }
}
