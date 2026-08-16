use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    Plan,
    Agent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    ReadOnly,
    MutationSafe,
    MutationDangerous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionDecision {
    pub permitted: bool,
    pub requires_approval: bool,
    pub reason: String,
    pub risk_level: RiskLevel,
}

pub struct PermissionPolicy {
    pub workspace_root: Option<PathBuf>,
    pub mode: ExecutionMode,
}

impl PermissionPolicy {
    pub fn new(workspace_root: Option<PathBuf>, mode: ExecutionMode) -> Self {
        Self {
            workspace_root,
            mode,
        }
    }

    pub fn is_safe_path(&self, target_path: &Path) -> bool {
        let root = match &self.workspace_root {
            Some(r) => match r.canonicalize() {
                Ok(c) => c,
                Err(_) => r.clone(),
            },
            None => return false,
        };

        let target = if target_path.is_absolute() {
            match target_path.canonicalize() {
                Ok(c) => c,
                Err(_) => target_path.to_path_buf(),
            }
        } else {
            match root.join(target_path).canonicalize() {
                Ok(c) => c,
                Err(_) => root.join(target_path),
            }
        };

        target.starts_with(&root)
    }

    pub fn evaluate_action(&self, action: &str, target_path: Option<&Path>, command: Option<&str>) -> PermissionDecision {
        // Path traversal check
        if let Some(path) = target_path {
            if !self.is_safe_path(path) {
                return PermissionDecision {
                    permitted: false,
                    requires_approval: true,
                    reason: format!("Path '{}' is outside the workspace root boundary.", path.display()),
                    risk_level: RiskLevel::MutationDangerous,
                };
            }
        }

        // Dangerous command detection
        if let Some(cmd) = command {
            let lower = cmd.to_lowercase();
            if lower.contains("rm -rf") || lower.contains("format ") || lower.contains("git push --force") || lower.contains("curl ") || lower.contains("wget ") {
                return PermissionDecision {
                    permitted: false,
                    requires_approval: true,
                    reason: format!("Command contains dangerous or network operations: '{}'", cmd),
                    risk_level: RiskLevel::MutationDangerous,
                };
            }
        }

        // Action classification
        let (risk, is_mutation) = match action {
            "read_file" | "list_directory" | "glob" | "grep" | "git_status" | "git_diff" => (RiskLevel::ReadOnly, false),
            "write_file" | "edit_file" | "git_stage" => (RiskLevel::MutationSafe, true),
            "git_commit" | "execute_command" => (RiskLevel::MutationDangerous, true),
            _ => (RiskLevel::MutationDangerous, true),
        };

        if self.mode == ExecutionMode::Plan && is_mutation {
            return PermissionDecision {
                permitted: false,
                requires_approval: true,
                reason: "Plan Mode is strictly read-only. File mutations and executions require user plan approval.".into(),
                risk_level: risk,
            };
        }

        if risk == RiskLevel::ReadOnly {
            PermissionDecision {
                permitted: true,
                requires_approval: false,
                reason: "Read-only workspace inspection is auto-approved.".into(),
                risk_level: risk,
            }
        } else {
            PermissionDecision {
                permitted: false,
                requires_approval: true,
                reason: format!("Action '{}' requires user approval.", action),
                risk_level: risk,
            }
        }
    }

    pub fn redact_secrets(input: &str) -> String {
        let mut text = input.to_string();
        // Redact typical API key patterns
        let patterns = [
            ("sk-[a-zA-Z0-9_-]{20,}", "[REDACTED_API_KEY]"),
            ("ghp_[a-zA-Z0-9]{20,}", "[REDACTED_GITHUB_TOKEN]"),
            ("bearer [a-zA-Z0-9_.-]{20,}", "Bearer [REDACTED_BEARER_TOKEN]"),
        ];

        for (pat, repl) in patterns {
            if let Ok(re) = regex_lite(pat) {
                text = re(text, repl);
            }
        }

        text
    }
}

fn regex_lite(pattern: &str) -> Result<Box<dyn Fn(String, &str) -> String>, ()> {
    // Simple substring replacement helper for common tokens
    if pattern.starts_with("sk-") {
        Ok(Box::new(|s: String, rep: &str| {
            if let Some(pos) = s.find("sk-") {
                let end = s[pos..].find(|c: char| c.is_whitespace() || c == '"' || c == '\'').map(|p| pos + p).unwrap_or(s.len());
                let mut out = s[..pos].to_string();
                out.push_str(rep);
                out.push_str(&s[end..]);
                out
            } else {
                s
            }
        }))
    } else {
        Ok(Box::new(|s: String, _| s))
    }
}
