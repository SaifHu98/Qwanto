use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use serde::{Deserialize, Serialize};
use crate::permission_policy::PermissionPolicy;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
    pub truncated: bool,
    pub command: Option<String>,
    pub cwd: Option<String>,
}

pub struct ToolExecutor;

impl ToolExecutor {
    const MAX_OUTPUT_BYTES: usize = 128 * 1024; // 128 KB limit
    const MAX_FILE_READ_BYTES: usize = 2 * 1024 * 1024; // 2 MB limit

    pub fn read_file(path_str: &str, policy: &PermissionPolicy) -> ToolResult {
        let path = Path::new(path_str);
        let decision = policy.evaluate_action("read_file", Some(path), None);
        if !decision.permitted {
            return ToolResult {
                success: false,
                output: String::new(),
                error: Some(decision.reason),
                truncated: false,
                command: None,
                cwd: None,
            };
        }

        let full_path = if path.is_absolute() {
            path.to_path_buf()
        } else if let Some(root) = &policy.workspace_root {
            root.join(path)
        } else {
            path.to_path_buf()
        };

        if !full_path.exists() {
            return ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("File does not exist: {}", full_path.display())),
                truncated: false,
                command: None,
                cwd: None,
            };
        }

        // Check file size
        if let Ok(meta) = fs::metadata(&full_path) {
            if meta.len() > Self::MAX_FILE_READ_BYTES as u64 {
                return ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("File size ({} bytes) exceeds 2MB safety limit.", meta.len())),
                    truncated: false,
                    command: None,
                    cwd: None,
                };
            }
        }

        match fs::read_to_string(&full_path) {
            Ok(content) => {
                let redacted = PermissionPolicy::redact_secrets(&content);
                ToolResult {
                    success: true,
                    output: redacted,
                    error: None,
                    truncated: false,
                    command: None,
                    cwd: None,
                }
            }
            Err(e) => ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to read file: {}", e)),
                truncated: false,
                command: None,
                cwd: None,
            },
        }
    }

    pub fn write_file(path_str: &str, content: &str, policy: &PermissionPolicy, approved: bool) -> ToolResult {
        let path = Path::new(path_str);
        let decision = policy.evaluate_action("write_file", Some(path), None);
        if !approved && !decision.permitted {
            return ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Action requires approval: {}", decision.reason)),
                truncated: false,
                command: None,
                cwd: None,
            };
        }

        let full_path = if path.is_absolute() {
            path.to_path_buf()
        } else if let Some(root) = &policy.workspace_root {
            root.join(path)
        } else {
            path.to_path_buf()
        };

        if let Some(parent) = full_path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        match fs::write(&full_path, content) {
            Ok(_) => ToolResult {
                success: true,
                output: format!("Successfully wrote {} bytes to {}", content.len(), full_path.display()),
                error: None,
                truncated: false,
                command: None,
                cwd: None,
            },
            Err(e) => ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to write file: {}", e)),
                truncated: false,
                command: None,
                cwd: None,
            },
        }
    }

    pub fn edit_file(path_str: &str, old_str: &str, new_str: &str, policy: &PermissionPolicy, approved: bool) -> ToolResult {
        let path = Path::new(path_str);
        let decision = policy.evaluate_action("edit_file", Some(path), None);
        if !approved && !decision.permitted {
            return ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Action requires approval: {}", decision.reason)),
                truncated: false,
                command: None,
                cwd: None,
            };
        }

        let full_path = if path.is_absolute() {
            path.to_path_buf()
        } else if let Some(root) = &policy.workspace_root {
            root.join(path)
        } else {
            path.to_path_buf()
        };

        match fs::read_to_string(&full_path) {
            Ok(content) => {
                if !content.contains(old_str) {
                    return ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some("Target substring not found in file.".into()),
                        truncated: false,
                        command: None,
                        cwd: None,
                    };
                }
                let updated = content.replacen(old_str, new_str, 1);
                match fs::write(&full_path, updated) {
                    Ok(_) => ToolResult {
                        success: true,
                        output: format!("Successfully applied edit to {}", full_path.display()),
                        error: None,
                        truncated: false,
                        command: None,
                        cwd: None,
                    },
                    Err(e) => ToolResult {
                        success: false,
                        output: String::new(),
                        error: Some(format!("Failed to write modified file: {}", e)),
                        truncated: false,
                        command: None,
                        cwd: None,
                    },
                }
            }
            Err(e) => ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to read file for editing: {}", e)),
                truncated: false,
                command: None,
                cwd: None,
            },
        }
    }

    pub fn list_directory(dir_str: Option<&str>, policy: &PermissionPolicy) -> ToolResult {
        let dir_path = match dir_str {
            Some(d) => PathBuf::from(d),
            None => policy.workspace_root.clone().unwrap_or_else(|| PathBuf::from(".")),
        };

        let decision = policy.evaluate_action("list_directory", Some(&dir_path), None);
        if !decision.permitted {
            return ToolResult {
                success: false,
                output: String::new(),
                error: Some(decision.reason),
                truncated: false,
                command: None,
                cwd: None,
            };
        }

        match fs::read_dir(&dir_path) {
            Ok(entries) => {
                let mut lines = Vec::new();
                for entry in entries.flatten() {
                    let file_name = entry.file_name().to_string_lossy().to_string();
                    let file_type = if entry.path().is_dir() { "DIR" } else { "FILE" };
                    lines.push(format!("[{}] {}", file_type, file_name));
                }
                lines.sort();
                ToolResult {
                    success: true,
                    output: lines.join("\n"),
                    error: None,
                    truncated: false,
                    command: None,
                    cwd: None,
                }
            }
            Err(e) => ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to list directory: {}", e)),
                truncated: false,
                command: None,
                cwd: None,
            },
        }
    }

    pub fn execute_command(command: &str, cwd: Option<&str>, policy: &PermissionPolicy, approved: bool) -> ToolResult {
        let decision = policy.evaluate_action("execute_command", None, Some(command));
        if !approved && !decision.permitted {
            return ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Command execution requires user approval: {}", decision.reason)),
                truncated: false,
                command: Some(command.to_string()),
                cwd: cwd.map(|s| s.to_string()),
            };
        }

        let work_dir = cwd
            .map(PathBuf::from)
            .or_else(|| policy.workspace_root.clone())
            .unwrap_or_else(|| PathBuf::from("."));

        #[cfg(target_os = "windows")]
        let mut cmd = Command::new("powershell");
        #[cfg(target_os = "windows")]
        cmd.arg("-NoProfile").arg("-Command").arg(command);

        #[cfg(not(target_os = "windows"))]
        let mut cmd = Command::new("sh");
        #[cfg(not(target_os = "windows"))]
        cmd.arg("-c").arg(command);

        cmd.current_dir(work_dir);

        match cmd.output() {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let combined = format!("{}{}", stdout, stderr);
                let redacted = PermissionPolicy::redact_secrets(&combined);

                let truncated = redacted.len() > Self::MAX_OUTPUT_BYTES;
                let final_output = if truncated {
                    format!("{}\n... [OUTPUT TRUNCATED AT 128KB]", &redacted[..Self::MAX_OUTPUT_BYTES])
                } else {
                    redacted
                };

                ToolResult {
                    success: output.status.success(),
                    output: final_output,
                    error: if output.status.success() { None } else { Some(format!("Command exited with code {:?}", output.status.code())) },
                    truncated,
                    command: Some(command.to_string()),
                    cwd: cwd.map(|s| s.to_string()),
                }
            }
            Err(e) => ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to spawn command: {}", e)),
                truncated: false,
                command: Some(command.to_string()),
                cwd: cwd.map(|s| s.to_string()),
            },
        }
    }
}
