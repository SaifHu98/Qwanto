use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use crate::permission_policy::PermissionPolicy;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    pub id: String,
    pub timestamp: String,
    pub step_type: String,
    pub content: String,
    pub tool_name: Option<String>,
    pub tool_args: Option<serde_json::Value>,
    pub tool_result: Option<serde_json::Value>,
    pub approval_status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSession {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub workspace_root: String,
    pub active_model: String,
    pub mode: String,
    pub steps: Vec<AgentStep>,
}

pub struct SessionStore {
    storage_dir: PathBuf,
}

impl SessionStore {
    pub fn new(custom_dir: Option<PathBuf>) -> Self {
        let storage_dir = custom_dir.unwrap_or_else(|| {
            #[cfg(target_os = "windows")]
            let base = std::env::var("APPDATA").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."));
            #[cfg(not(target_os = "windows"))]
            let base = std::env::var("HOME").map(|h| PathBuf::from(h).join(".local/share")).unwrap_or_else(|_| PathBuf::from("."));

            base.join("qwanto").join("sessions")
        });

        let _ = fs::create_dir_all(&storage_dir);
        Self { storage_dir }
    }

    pub fn validate_session_id(session_id: &str) -> Result<(), String> {
        if session_id.is_empty() || session_id.len() > 64 {
            return Err("Invalid session ID length.".into());
        }
        if !session_id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            return Err("Invalid session ID format: only alphanumeric, '-', and '_' are allowed.".into());
        }
        Ok(())
    }

    pub fn list_sessions(&self) -> Vec<AgentSession> {
        let mut list = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.storage_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("json") {
                    if let Ok(content) = fs::read_to_string(&path) {
                        if let Ok(session) = serde_json::from_str::<AgentSession>(&content) {
                            list.push(session);
                        }
                    }
                }
            }
        }
        list.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        list
    }

    pub fn save_session(&self, session: &AgentSession) -> Result<(), String> {
        Self::validate_session_id(&session.id)?;

        let mut sanitized = session.clone();
        sanitized.title = PermissionPolicy::redact_secrets(&sanitized.title);
        for step in &mut sanitized.steps {
            step.content = PermissionPolicy::redact_secrets(&step.content);
        }

        let file_path = self.storage_dir.join(format!("{}.json", sanitized.id));
        let tmp_path = self.storage_dir.join(format!("{}.tmp.{}", sanitized.id, std::process::id()));

        let json = serde_json::to_string_pretty(&sanitized).map_err(|e| e.to_string())?;

        // Atomic write
        {
            let mut f = File::create(&tmp_path).map_err(|e| e.to_string())?;
            f.write_all(json.as_bytes()).map_err(|e| e.to_string())?;
            f.sync_all().map_err(|e| e.to_string())?;
        }

        fs::rename(&tmp_path, &file_path).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn get_session(&self, session_id: &str) -> Option<AgentSession> {
        if Self::validate_session_id(session_id).is_err() {
            return None;
        }

        let file_path = self.storage_dir.join(format!("{}.json", session_id));
        if file_path.exists() {
            if let Ok(content) = fs::read_to_string(file_path) {
                return serde_json::from_str(&content).ok();
            }
        }
        None
    }

    pub fn delete_session(&self, session_id: &str) -> Result<(), String> {
        Self::validate_session_id(session_id)?;
        let file_path = self.storage_dir.join(format!("{}.json", session_id));
        if file_path.exists() {
            fs::remove_file(file_path).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
