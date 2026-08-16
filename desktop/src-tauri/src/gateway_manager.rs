use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

const READY_PREFIX: &str = "QWANTO_GATEWAY_READY ";

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
            .env("QWANTO_MODEL_ROOT", &model_root)
            .env("QWANTO_MODEL_PATHS", &model_root);
        if let Some(qwnrun) = qwnrun {
            command
                .arg("--engine")
                .arg(&qwnrun)
                .env("QWANTO_QWNRUN", &qwnrun);
        }
        command.stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut child = command.spawn().map_err(|error| self.fail(format!("Failed to start gateway sidecar: {error}")))?;
        let stdout = child.stdout.take().ok_or_else(|| self.fail("Gateway stdout was not piped.".into()))?;
        if let Some(stderr) = child.stderr.take() {
            std::thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for _line in reader.lines().map_while(Result::ok) {}
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
            self.status.error = Some(format!("Gateway exited with {exit}"));
            self.child = None;
        }
        self.status.clone()
    }

    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            #[cfg(windows)]
            {
                let pid = child.id().to_string();
                let _ = Command::new("taskkill").args(["/PID", &pid, "/T", "/F"]).status();
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
}
