use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use serde::{Deserialize, Serialize};
use crate::permission_policy::{ActionDetails, PermissionPolicy, PolicyOutcome};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub outcome: String, // "executed", "needs_approval", "denied"
    pub output: String,
    pub error: Option<String>,
    pub truncated: bool,
    pub approval_token: Option<String>,
    pub action_details: Option<ActionDetails>,
}

pub struct ToolExecutor;

impl ToolExecutor {
    const MAX_OUTPUT_BYTES: usize = 128 * 1024; // 128 KB
    const MAX_FILE_READ_BYTES: usize = 2 * 1024 * 1024; // 2 MB

    pub fn read_file(session_id: &str, path_str: &str, policy: &PermissionPolicy) -> ToolResult {
        let path = Path::new(path_str);
        let args_hash = format!("read:{}", path_str);
        let outcome = policy.evaluate_action(session_id, "read_file", Some(path), None, &args_hash, None);

        match outcome {
            PolicyOutcome::Deny { reason } => ToolResult {
                success: false,
                outcome: "denied".into(),
                output: String::new(),
                error: Some(reason),
                truncated: false,
                approval_token: None,
                action_details: None,
            },
            PolicyOutcome::NeedsApproval { token, details } => ToolResult {
                success: false,
                outcome: "needs_approval".into(),
                output: String::new(),
                error: None,
                truncated: false,
                approval_token: Some(token),
                action_details: Some(details),
            },
            PolicyOutcome::Allow => {
                let validated_path = match policy.validate_path(path) {
                    Ok(p) => p,
                    Err(err) => return ToolResult {
                        success: false,
                        outcome: "denied".into(),
                        output: String::new(),
                        error: Some(err),
                        truncated: false,
                        approval_token: None,
                        action_details: None,
                    },
                };

                if !validated_path.exists() {
                    return ToolResult {
                        success: false,
                        outcome: "error".into(),
                        output: String::new(),
                        error: Some(format!("File not found: {}", validated_path.display())),
                        truncated: false,
                        approval_token: None,
                        action_details: None,
                    };
                }

                if let Ok(meta) = fs::metadata(&validated_path) {
                    if meta.len() > Self::MAX_FILE_READ_BYTES as u64 {
                        return ToolResult {
                            success: false,
                            outcome: "error".into(),
                            output: String::new(),
                            error: Some(format!("File exceeds 2MB read limit ({} bytes)", meta.len())),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        };
                    }
                }

                match fs::read_to_string(&validated_path) {
                    Ok(content) => {
                        let redacted = PermissionPolicy::redact_secrets(&content);
                        ToolResult {
                            success: true,
                            outcome: "executed".into(),
                            output: redacted,
                            error: None,
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        }
                    }
                    Err(e) => ToolResult {
                        success: false,
                        outcome: "error".into(),
                        output: String::new(),
                        error: Some(format!("Failed to read file: {}", e)),
                        truncated: false,
                        approval_token: None,
                        action_details: None,
                    },
                }
            }
        }
    }

    pub fn write_file(
        session_id: &str,
        path_str: &str,
        content: &str,
        approval_token: Option<&str>,
        policy: &PermissionPolicy,
    ) -> ToolResult {
        let path = Path::new(path_str);
        let args_hash = format!("write:{}:{}:len={}", path_str, content.len(), content.lines().count());
        let outcome = policy.evaluate_action(session_id, "write_file", Some(path), None, &args_hash, Some(content));

        match outcome {
            PolicyOutcome::Deny { reason } => ToolResult {
                success: false,
                outcome: "denied".into(),
                output: String::new(),
                error: Some(reason),
                truncated: false,
                approval_token: None,
                action_details: None,
            },
            PolicyOutcome::NeedsApproval { token, details } => {
                // If caller provided token, verify and consume it
                if let Some(tok) = approval_token {
                    let root = match &policy.workspace_root {
                        Some(r) => r,
                        None => return ToolResult {
                            success: false,
                            outcome: "denied".into(),
                            output: String::new(),
                            error: Some("Workspace root is not set.".into()),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        },
                    };

                    if let Err(auth_err) = policy.token_registry.consume_token(
                        tok,
                        session_id,
                        "write_file",
                        &args_hash,
                        root,
                        policy.mode,
                    ) {
                        return ToolResult {
                            success: false,
                            outcome: "denied".into(),
                            output: String::new(),
                            error: Some(format!("Authorization Denied: {}", auth_err)),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        };
                    }

                    // Approved and authorized: execute write
                    let validated_path = match policy.validate_path(path) {
                        Ok(p) => p,
                        Err(err) => return ToolResult {
                            success: false,
                            outcome: "denied".into(),
                            output: String::new(),
                            error: Some(err),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        },
                    };

                    if let Some(parent) = validated_path.parent() {
                        let _ = fs::create_dir_all(parent);
                    }

                    match fs::write(&validated_path, content) {
                        Ok(_) => ToolResult {
                            success: true,
                            outcome: "executed".into(),
                            output: format!("Successfully wrote {} bytes to {}", content.len(), validated_path.display()),
                            error: None,
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        },
                        Err(e) => ToolResult {
                            success: false,
                            outcome: "error".into(),
                            output: String::new(),
                            error: Some(format!("Write failed: {}", e)),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        },
                    }
                } else {
                    // Return token for UI approval prompt
                    ToolResult {
                        success: false,
                        outcome: "needs_approval".into(),
                        output: String::new(),
                        error: None,
                        truncated: false,
                        approval_token: Some(token),
                        action_details: Some(details),
                    }
                }
            }
            PolicyOutcome::Allow => unreachable!(),
        }
    }

    pub fn edit_file(
        session_id: &str,
        path_str: &str,
        old_str: &str,
        new_str: &str,
        approval_token: Option<&str>,
        policy: &PermissionPolicy,
    ) -> ToolResult {
        let path = Path::new(path_str);
        let diff_preview = format!("- {}\n+ {}", old_str.trim(), new_str.trim());
        let args_hash = format!("edit:{}:old_len={}:new_len={}", path_str, old_str.len(), new_str.len());
        let outcome = policy.evaluate_action(session_id, "edit_file", Some(path), None, &args_hash, Some(&diff_preview));

        match outcome {
            PolicyOutcome::Deny { reason } => ToolResult {
                success: false,
                outcome: "denied".into(),
                output: String::new(),
                error: Some(reason),
                truncated: false,
                approval_token: None,
                action_details: None,
            },
            PolicyOutcome::NeedsApproval { token, details } => {
                if let Some(tok) = approval_token {
                    let root = match &policy.workspace_root {
                        Some(r) => r,
                        None => return ToolResult {
                            success: false,
                            outcome: "denied".into(),
                            output: String::new(),
                            error: Some("Workspace root is not set.".into()),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        },
                    };

                    if let Err(auth_err) = policy.token_registry.consume_token(
                        tok,
                        session_id,
                        "edit_file",
                        &args_hash,
                        root,
                        policy.mode,
                    ) {
                        return ToolResult {
                            success: false,
                            outcome: "denied".into(),
                            output: String::new(),
                            error: Some(format!("Authorization Denied: {}", auth_err)),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        };
                    }

                    let validated_path = match policy.validate_path(path) {
                        Ok(p) => p,
                        Err(err) => return ToolResult {
                            success: false,
                            outcome: "denied".into(),
                            output: String::new(),
                            error: Some(err),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        },
                    };

                    let content = match fs::read_to_string(&validated_path) {
                        Ok(c) => c,
                        Err(e) => return ToolResult {
                            success: false,
                            outcome: "error".into(),
                            output: String::new(),
                            error: Some(format!("Failed to read target file: {}", e)),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        },
                    };

                    if !content.contains(old_str) {
                        return ToolResult {
                            success: false,
                            outcome: "error".into(),
                            output: String::new(),
                            error: Some("Target substring not found in file.".into()),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        };
                    }

                    let updated = content.replacen(old_str, new_str, 1);
                    match fs::write(&validated_path, updated) {
                        Ok(_) => ToolResult {
                            success: true,
                            outcome: "executed".into(),
                            output: format!("Successfully applied edit to {}", validated_path.display()),
                            error: None,
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        },
                        Err(e) => ToolResult {
                            success: false,
                            outcome: "error".into(),
                            output: String::new(),
                            error: Some(format!("Failed to write modifications: {}", e)),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        },
                    }
                } else {
                    ToolResult {
                        success: false,
                        outcome: "needs_approval".into(),
                        output: String::new(),
                        error: None,
                        truncated: false,
                        approval_token: Some(token),
                        action_details: Some(details),
                    }
                }
            }
            PolicyOutcome::Allow => unreachable!(),
        }
    }

    pub fn list_directory(session_id: &str, dir_opt: Option<&str>, policy: &PermissionPolicy) -> ToolResult {
        let dir_path = match dir_opt {
            Some(d) => Path::new(d),
            None => match &policy.workspace_root {
                Some(r) => r.as_path(),
                None => return ToolResult {
                    success: false,
                    outcome: "denied".into(),
                    output: String::new(),
                    error: Some("Workspace root is not configured.".into()),
                    truncated: false,
                    approval_token: None,
                    action_details: None,
                },
            },
        };

        let args_hash = format!("list:{}", dir_path.display());
        let outcome = policy.evaluate_action(session_id, "list_directory", Some(dir_path), None, &args_hash, None);

        match outcome {
            PolicyOutcome::Deny { reason } => ToolResult {
                success: false,
                outcome: "denied".into(),
                output: String::new(),
                error: Some(reason),
                truncated: false,
                approval_token: None,
                action_details: None,
            },
            PolicyOutcome::NeedsApproval { token, details } => ToolResult {
                success: false,
                outcome: "needs_approval".into(),
                output: String::new(),
                error: None,
                truncated: false,
                approval_token: Some(token),
                action_details: Some(details),
            },
            PolicyOutcome::Allow => {
                let validated_path = match policy.validate_path(dir_path) {
                    Ok(p) => p,
                    Err(err) => return ToolResult {
                        success: false,
                        outcome: "denied".into(),
                        output: String::new(),
                        error: Some(err),
                        truncated: false,
                        approval_token: None,
                        action_details: None,
                    },
                };

                match fs::read_dir(&validated_path) {
                    Ok(entries) => {
                        let mut items = Vec::new();
                        for entry in entries.flatten() {
                            let name = entry.file_name().to_string_lossy().to_string();
                            let is_dir = entry.path().is_dir();
                            items.push(format!("[{}] {}", if is_dir { "DIR" } else { "FILE" }, name));
                        }
                        items.sort();
                        ToolResult {
                            success: true,
                            outcome: "executed".into(),
                            output: items.join("\n"),
                            error: None,
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        }
                    }
                    Err(e) => ToolResult {
                        success: false,
                        outcome: "error".into(),
                        output: String::new(),
                        error: Some(format!("Failed to read directory: {}", e)),
                        truncated: false,
                        approval_token: None,
                        action_details: None,
                    },
                }
            }
        }
    }

    pub fn execute_command(
        session_id: &str,
        program: &str,
        args: Vec<String>,
        cwd_opt: Option<&str>,
        approval_token: Option<&str>,
        policy: &PermissionPolicy,
    ) -> ToolResult {
        let full_cmd_str = format!("{} {}", program, args.join(" "));
        let args_hash = format!("cmd:{}:{}", program, args.join(","));
        let outcome = policy.evaluate_action(
            session_id,
            "execute_command",
            None,
            Some(&full_cmd_str),
            &args_hash,
            Some(&full_cmd_str),
        );

        match outcome {
            PolicyOutcome::Deny { reason } => ToolResult {
                success: false,
                outcome: "denied".into(),
                output: String::new(),
                error: Some(reason),
                truncated: false,
                approval_token: None,
                action_details: None,
            },
            PolicyOutcome::NeedsApproval { token, details } => {
                if let Some(tok) = approval_token {
                    let root = match &policy.workspace_root {
                        Some(r) => r,
                        None => return ToolResult {
                            success: false,
                            outcome: "denied".into(),
                            output: String::new(),
                            error: Some("Workspace root is not configured.".into()),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        },
                    };

                    if let Err(auth_err) = policy.token_registry.consume_token(
                        tok,
                        session_id,
                        "execute_command",
                        &args_hash,
                        root,
                        policy.mode,
                    ) {
                        return ToolResult {
                            success: false,
                            outcome: "denied".into(),
                            output: String::new(),
                            error: Some(format!("Authorization Denied: {}", auth_err)),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        };
                    }

                    // Direct structured command execution (no shell!)
                    let cwd_path = cwd_opt.map(Path::new);
                    let validated_cwd = match policy.validate_cwd(cwd_path) {
                        Ok(p) => p,
                        Err(err) => return ToolResult {
                            success: false,
                            outcome: "denied".into(),
                            output: String::new(),
                            error: Some(err),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        },
                    };

                    let mut cmd = Command::new(program);
                    cmd.args(&args);
                    cmd.current_dir(validated_cwd);
                    cmd.stdout(Stdio::piped());
                    cmd.stderr(Stdio::piped());

                    match cmd.output() {
                        Ok(output) => {
                            let stdout = String::from_utf8_lossy(&output.stdout);
                            let stderr = String::from_utf8_lossy(&output.stderr);
                            let combined = format!("{}{}", stdout, stderr);
                            let redacted = PermissionPolicy::redact_secrets(&combined);

                            let truncated = redacted.len() > Self::MAX_OUTPUT_BYTES;
                            let final_output = if truncated {
                                format!("{}\n... [TRUNCATED AT 128KB]", &redacted[..Self::MAX_OUTPUT_BYTES])
                            } else {
                                redacted
                            };

                            ToolResult {
                                success: output.status.success(),
                                outcome: "executed".into(),
                                output: final_output,
                                error: if output.status.success() {
                                    None
                                } else {
                                    Some(format!("Command exited with status code {:?}", output.status.code()))
                                },
                                truncated,
                                approval_token: None,
                                action_details: None,
                            }
                        }
                        Err(e) => ToolResult {
                            success: false,
                            outcome: "error".into(),
                            output: String::new(),
                            error: Some(format!("Failed to spawn structured process '{}': {}", program, e)),
                            truncated: false,
                            approval_token: None,
                            action_details: None,
                        },
                    }
                } else {
                    ToolResult {
                        success: false,
                        outcome: "needs_approval".into(),
                        output: String::new(),
                        error: None,
                        truncated: false,
                        approval_token: Some(token),
                        action_details: Some(details),
                    }
                }
            }
            PolicyOutcome::Allow => unreachable!(),
        }
    }
}
