use std::fs::{self, File};
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStep {
    pub id: String,
    pub timestamp: String,
    pub step_type: String, // "plan", "tool_call", "tool_result", "user_message", "assistant_message"
    pub content: String,
    pub tool_name: Option<String>,
    pub tool_args: Option<serde_json::Value>,
    pub tool_result: Option<serde_json::Value>,
    pub approval_status: Option<String>, // "pending", "approved", "rejected"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSession {
    pub id: String,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub workspace_root: String,
    pub active_model: String,
    pub mode: String, // "plan", "agent"
    pub steps: Vec<AgentStep>,
}

pub struct SessionStore {
    storage_dir: PathBuf,
}

impl SessionStore {
    pub fn new(custom_dir: Option<PathBuf>) -> Self {
        let storage_dir = custom_dir.unwrap_or_else(|| {
            PathBuf::from("D:/EcoUni/qwanto/.qwanto/sessions")
        });

        let _ = fs::create_dir_all(&storage_dir);
        Self { storage_dir }
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
        let file_path = self.storage_dir.join(format!("{}.json", session.id));
        let json = serde_json::to_string_pretty(session).map_err(|e| e.to_string())?;
        fs::write(file_path, json).map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn get_session(&self, session_id: &str) -> Option<AgentSession> {
        let file_path = self.storage_dir.join(format!("{}.json", session_id));
        if file_path.exists() {
            if let Ok(content) = fs::read_to_string(file_path) {
                return serde_json::from_str(&content).ok();
            }
        }
        None
    }

    pub fn delete_session(&self, session_id: &str) -> Result<(), String> {
        let file_path = self.storage_dir.join(format!("{}.json", session_id));
        if file_path.exists() {
            fs::remove_file(file_path).map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
