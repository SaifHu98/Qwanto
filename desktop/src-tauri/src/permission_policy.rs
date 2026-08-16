use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    Plan,
    Agent,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActionDetails {
    pub tool_name: String,
    pub description: String,
    pub target_path: Option<String>,
    pub command: Option<String>,
    pub diff_preview: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PolicyOutcome {
    Allow,
    NeedsApproval {
        token: String,
        details: ActionDetails,
    },
    Deny {
        reason: String,
    },
}

#[derive(Debug, Clone)]
pub struct ApprovalToken {
    pub token_id: String,
    pub session_id: String,
    pub tool_name: String,
    pub args_hash: String,
    pub canonical_workspace: PathBuf,
    pub mode: ExecutionMode,
    pub created_at: Instant,
    pub ttl: Duration,
}

pub struct ApprovalTokenRegistry {
    tokens: Mutex<HashMap<String, ApprovalToken>>,
}

impl ApprovalTokenRegistry {
    pub fn new() -> Self {
        Self {
            tokens: Mutex::new(HashMap::new()),
        }
    }

    pub fn issue_token(
        &self,
        session_id: &str,
        tool_name: &str,
        args_hash: &str,
        canonical_workspace: &Path,
        mode: ExecutionMode,
    ) -> String {
        let mut guard = self.tokens.lock().unwrap();
        let now = Instant::now();
        guard.retain(|_, tok| now.duration_since(tok.created_at) < tok.ttl);

        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
        let rand_val = pseudo_secure_seed();
        let token_id = format!("tok_{:x}_{:x}_{:x}", ts, rand_val, guard.len());

        let token = ApprovalToken {
            token_id: token_id.clone(),
            session_id: session_id.to_string(),
            tool_name: tool_name.to_string(),
            args_hash: args_hash.to_string(),
            canonical_workspace: canonical_workspace.to_path_buf(),
            mode,
            created_at: now,
            ttl: Duration::from_secs(300), // 5 minutes TTL
        };

        guard.insert(token_id.clone(), token);
        token_id
    }

    pub fn consume_token(
        &self,
        token_id: &str,
        session_id: &str,
        tool_name: &str,
        args_hash: &str,
        canonical_workspace: &Path,
        current_mode: ExecutionMode,
    ) -> Result<(), String> {
        let mut guard = self.tokens.lock().unwrap();
        let token = match guard.remove(token_id) {
            Some(tok) => tok,
            None => return Err("Invalid, already-consumed, or nonexistent approval token.".into()),
        };

        if Instant::now().duration_since(token.created_at) >= token.ttl {
            return Err("Approval token has expired.".into());
        }

        if current_mode == ExecutionMode::Plan || token.mode == ExecutionMode::Plan {
            return Err("Plan Mode strictly forbids mutations even with an approval token. Switch to Agent Mode first.".into());
        }

        if token.session_id != session_id {
            return Err("Approval token session mismatch.".into());
        }
        if token.tool_name != tool_name {
            return Err("Approval token tool mismatch.".into());
        }
        if token.args_hash != args_hash {
            return Err("Approval token argument mismatch: arguments were modified after approval.".into());
        }
        if token.canonical_workspace != canonical_workspace {
            return Err("Approval token workspace root mismatch.".into());
        }

        Ok(())
    }
}

pub struct PermissionPolicy {
    pub workspace_root: Option<PathBuf>,
    pub mode: ExecutionMode,
    pub token_registry: ApprovalTokenRegistry,
}

impl PermissionPolicy {
    pub fn new(workspace_root: Option<PathBuf>, mode: ExecutionMode) -> Self {
        let canonical_root = workspace_root.and_then(|r| r.canonicalize().ok());
        Self {
            workspace_root: canonical_root,
            mode,
            token_registry: ApprovalTokenRegistry::new(),
        }
    }

    pub fn set_workspace_root(&mut self, root: &Path) -> Result<PathBuf, String> {
        if !root.exists() || !root.is_dir() {
            return Err(format!("Workspace path does not exist or is not a directory: {}", root.display()));
        }
        let canonical = root.canonicalize().map_err(|e| format!("Failed to canonicalize workspace: {}", e))?;
        self.workspace_root = Some(canonical.clone());
        Ok(canonical)
    }

    pub fn validate_path(&self, target: &Path) -> Result<PathBuf, String> {
        let root = match &self.workspace_root {
            Some(r) => r,
            None => return Err("No workspace root has been configured. File operations are blocked.".into()),
        };

        let resolved = if target.is_absolute() {
            target.to_path_buf()
        } else {
            root.join(target)
        };

        if resolved.exists() {
            let canonical = resolved.canonicalize().map_err(|e| format!("Path canonicalization failed: {}", e))?;
            if !canonical.starts_with(root) {
                return Err(format!("Security Violation: Path '{}' traverses outside the workspace boundary.", target.display()));
            }
            Ok(canonical)
        } else {
            let mut ancestor = resolved.clone();
            while let Some(parent) = ancestor.parent() {
                if parent.exists() {
                    let canonical_parent = parent.canonicalize().map_err(|e| format!("Ancestor canonicalization failed: {}", e))?;
                    if !canonical_parent.starts_with(root) {
                        return Err(format!("Security Violation: Target path '{}' creates files outside workspace boundary.", target.display()));
                    }
                    break;
                }
                ancestor = parent.to_path_buf();
            }
            let path_str = resolved.to_string_lossy();
            if path_str.contains("..") {
                return Err(format!("Security Violation: Path contains forbidden relative traversal '..': {}", target.display()));
            }
            Ok(resolved)
        }
    }

    pub fn validate_cwd(&self, cwd_opt: Option<&Path>) -> Result<PathBuf, String> {
        let root = match &self.workspace_root {
            Some(r) => r,
            None => return Err("No workspace root configured. Subprocess execution blocked.".into()),
        };

        match cwd_opt {
            Some(cwd) => {
                if !cwd.exists() || !cwd.is_dir() {
                    return Err(format!("Working directory does not exist: {}", cwd.display()));
                }
                let canonical = cwd.canonicalize().map_err(|e| format!("CWD canonicalization failed: {}", e))?;
                if !canonical.starts_with(root) {
                    return Err(format!("Security Violation: Working directory '{}' is outside workspace boundary.", cwd.display()));
                }
                Ok(canonical)
            }
            None => Ok(root.clone()),
        }
    }

    pub fn evaluate_action(
        &self,
        session_id: &str,
        tool_name: &str,
        target_path: Option<&Path>,
        command: Option<&str>,
        args_hash: &str,
        diff_preview: Option<&str>,
    ) -> PolicyOutcome {
        let root = match &self.workspace_root {
            Some(r) => r,
            None => return PolicyOutcome::Deny {
                reason: "No workspace root configured. All operations are strictly denied.".into(),
            },
        };

        if let Some(path) = target_path {
            if let Err(err) = self.validate_path(path) {
                return PolicyOutcome::Deny { reason: err };
            }
        }

        if let Some(cmd) = command {
            let lower = cmd.to_lowercase();
            if cmd.contains('|') || cmd.contains(';') || cmd.contains('&') || cmd.contains('`') || cmd.contains('$') || cmd.contains('>') || cmd.contains('<') {
                return PolicyOutcome::Deny {
                    reason: "Security Violation: Command contains forbidden shell metacharacters (pipes, redirection, chaining). Use structured tools.".into(),
                };
            }
            let network_bins = ["curl", "wget", "nc", "netcat", "ssh", "scp", "ftp", "telnet", "git push --force"];
            for bin in network_bins {
                if lower.starts_with(bin) || lower.contains(&format!(" {}", bin)) {
                    return PolicyOutcome::Deny {
                        reason: format!("Security Violation: Network-capable or force command '{}' is strictly blocked in local-only mode.", bin),
                    };
                }
            }
        }

        let is_mutation = match tool_name {
            "read_file" | "list_directory" | "glob" | "grep" | "git_status" | "git_diff" => false,
            "write_file" | "edit_file" | "git_stage" | "git_commit" | "execute_command" => true,
            _ => true,
        };

        if self.mode == ExecutionMode::Plan && is_mutation {
            return PolicyOutcome::Deny {
                reason: "Plan Mode is strictly read-only. Mutations cannot be executed or approved in Plan Mode. Switch to Agent Mode first.".into(),
            };
        }

        if !is_mutation {
            return PolicyOutcome::Allow;
        }

        let token = self.token_registry.issue_token(session_id, tool_name, args_hash, root, self.mode);
        let description = match tool_name {
            "write_file" => format!("Write file: {}", target_path.map(|p| p.display().to_string()).unwrap_or_default()),
            "edit_file" => format!("Apply edit: {}", target_path.map(|p| p.display().to_string()).unwrap_or_default()),
            "git_commit" => "Create Git commit".to_string(),
            "execute_command" => format!("Run command: {}", command.unwrap_or_default()),
            _ => format!("Execute tool '{}'", tool_name),
        };

        PolicyOutcome::NeedsApproval {
            token,
            details: ActionDetails {
                tool_name: tool_name.to_string(),
                description,
                target_path: target_path.map(|p| p.display().to_string()),
                command: command.map(|s| s.to_string()),
                diff_preview: diff_preview.map(|s| s.to_string()),
            },
        }
    }

    pub fn redact_secrets(input: &str) -> String {
        let mut text = input.to_string();

        text = redact_pattern(&text, "sk-", |c| c.is_alphanumeric() || c == '_' || c == '-', 20, "[REDACTED_API_KEY]");
        text = redact_pattern(&text, "ghp_", |c| c.is_alphanumeric(), 20, "[REDACTED_GITHUB_TOKEN]");
        text = redact_pattern(&text, "github_pat_", |c| c.is_alphanumeric() || c == '_', 20, "[REDACTED_GITHUB_PAT]");
        text = redact_pattern(&text, "Bearer ", |c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.', 20, "Bearer [REDACTED_BEARER_TOKEN]");
        text = redact_pattern(&text, "bearer ", |c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.', 20, "bearer [REDACTED_BEARER_TOKEN]");

        while let Some(start) = text.find("-----BEGIN") {
            if let Some(end) = text[start..].find("-----END") {
                if let Some(final_dashes) = text[start + end..].find("-----") {
                    let end_pos = start + end + final_dashes + 5;
                    text = format!("{}[REDACTED_PRIVATE_KEY]{}", &text[..start], &text[end_pos..]);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        let db_schemes = ["postgres://", "postgresql://", "mysql://", "mongodb://", "redis://"];
        for scheme in db_schemes {
            while let Some(start) = text.find(scheme) {
                let rest = &text[start + scheme.len()..];
                if let Some(at_pos) = rest.find('@') {
                    let colon_pos = rest[..at_pos].find(':');
                    if let Some(cp) = colon_pos {
                        let password_start = start + scheme.len() + cp + 1;
                        let password_end = start + scheme.len() + at_pos;
                        text = format!("{}[REDACTED_DB_PASSWORD]{}", &text[..password_start], &text[password_end..]);
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        text
    }
}

fn redact_pattern<F>(text: &str, prefix: &str, is_valid_char: F, min_len: usize, replacement: &str) -> String
where
    F: Fn(char) -> bool,
{
    let mut result = String::new();
    let mut cursor = 0;

    while let Some(pos) = text[cursor..].find(prefix) {
        let abs_pos = cursor + pos;
        result.push_str(&text[cursor..abs_pos]);

        let token_start = abs_pos + prefix.len();
        let token_len = text[token_start..].chars().take_while(|&c| is_valid_char(c)).count();

        if token_len >= min_len {
            result.push_str(replacement);
            cursor = token_start + token_len;
        } else {
            result.push_str(&text[abs_pos..token_start]);
            cursor = token_start;
        }
    }

    result.push_str(&text[cursor..]);
    result
}

fn pseudo_secure_seed() -> u64 {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;
    let ptr = &nanos as *const _ as usize as u64;
    nanos.wrapping_mul(0x517cc1b727220a95).wrapping_add(ptr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_plan_mode_hard_denial_on_mutation() {
        let temp_dir = std::env::temp_dir().join("qwanto_test_plan_denial");
        let _ = fs::create_dir_all(&temp_dir);
        let policy = PermissionPolicy::new(Some(temp_dir.clone()), ExecutionMode::Plan);

        let outcome = policy.evaluate_action(
            "sess-1",
            "write_file",
            Some(&temp_dir.join("test.txt")),
            None,
            "hash123",
            None,
        );

        match outcome {
            PolicyOutcome::Deny { reason } => {
                assert!(reason.contains("Plan Mode is strictly read-only"));
            }
            _ => panic!("Plan mode must hard-deny write_file"),
        }
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_token_reuse_fails() {
        let registry = ApprovalTokenRegistry::new();
        let root = Path::new("/workspace");
        let token = registry.issue_token("sess-1", "write_file", "hash1", root, ExecutionMode::Agent);

        // First consume succeeds
        assert!(registry.consume_token(&token, "sess-1", "write_file", "hash1", root, ExecutionMode::Agent).is_ok());

        // Reusing the same token must immediately fail
        let reuse_res = registry.consume_token(&token, "sess-1", "write_file", "hash1", root, ExecutionMode::Agent);
        assert!(reuse_res.is_err());
    }

    #[test]
    fn test_token_args_mutation_fails() {
        let registry = ApprovalTokenRegistry::new();
        let root = Path::new("/workspace");
        let token = registry.issue_token("sess-1", "write_file", "original_hash", root, ExecutionMode::Agent);

        // Attempting to consume with altered arguments must fail
        let mutated_res = registry.consume_token(&token, "sess-1", "write_file", "tampered_hash", root, ExecutionMode::Agent);
        assert!(mutated_res.is_err());
    }

    #[test]
    fn test_secret_redaction() {
        let text = "Keys: sk-abcdef1234567890abcdef and ghp_1234567890abcdef123456";
        let redacted = PermissionPolicy::redact_secrets(text);
        assert!(!redacted.contains("sk-abcdef1234567890abcdef"));
        assert!(redacted.contains("[REDACTED_API_KEY]"));
        assert!(!redacted.contains("ghp_1234567890abcdef123456"));
        assert!(redacted.contains("[REDACTED_GITHUB_TOKEN]"));
    }
}
