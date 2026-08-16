use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectMemory {
    pub schema_version: u32,
    pub workspace_root: String,
    pub enabled: bool,
    pub summary: String,
    pub architecture_notes: String,
    pub user_conventions: String,
    pub accepted_decisions: Vec<String>,
    pub task_checkpoints: Vec<String>,
    pub updated_at: String,
}

impl ProjectMemory {
    pub fn for_workspace(root: &Path) -> Self {
        Self {
            schema_version: 1,
            workspace_root: root.to_string_lossy().to_string(),
            enabled: true,
            summary: String::new(),
            architecture_notes: String::new(),
            user_conventions: String::new(),
            accepted_decisions: Vec::new(),
            task_checkpoints: Vec::new(),
            updated_at: timestamp(),
        }
    }
}

pub struct ProjectMemoryStore;

impl ProjectMemoryStore {
    fn path(root: &Path) -> PathBuf {
        root.join(".qwanto").join("project-memory.json")
    }

    pub fn load(root: &Path) -> Result<ProjectMemory, String> {
        let path = Self::path(root);
        if !path.is_file() {
            return Ok(ProjectMemory::for_workspace(root));
        }
        let content = fs::read_to_string(&path).map_err(|error| error.to_string())?;
        let mut memory: ProjectMemory = serde_json::from_str(&content).map_err(|error| error.to_string())?;
        memory.workspace_root = root.to_string_lossy().to_string();
        Ok(memory)
    }

    pub fn save(root: &Path, mut memory: ProjectMemory) -> Result<ProjectMemory, String> {
        let directory = root.join(".qwanto");
        fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        memory.schema_version = 1;
        memory.workspace_root = root.to_string_lossy().to_string();
        memory.updated_at = timestamp();
        let path = Self::path(root);
        let temp_path = directory.join(format!("project-memory.json.tmp.{}", std::process::id()));
        let json = serde_json::to_string_pretty(&memory).map_err(|error| error.to_string())?;
        {
            let mut file = File::create(&temp_path).map_err(|error| error.to_string())?;
            file.write_all(json.as_bytes()).map_err(|error| error.to_string())?;
            file.sync_all().map_err(|error| error.to_string())?;
        }
        #[cfg(windows)]
        if path.exists() {
            fs::remove_file(&path).map_err(|error| error.to_string())?;
        }
        fs::rename(&temp_path, &path).map_err(|error| error.to_string())?;
        Ok(memory)
    }

    pub fn clear(root: &Path) -> Result<ProjectMemory, String> {
        let path = Self::path(root);
        if path.exists() {
            fs::remove_file(path).map_err(|error| error.to_string())?;
        }
        Ok(ProjectMemory::for_workspace(root))
    }

    pub fn export(root: &Path) -> Result<String, String> {
        let memory = Self::load(root)?;
        serde_json::to_string_pretty(&memory).map_err(|error| error.to_string())
    }
}

fn timestamp() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_memory_round_trips_atomically() {
        let root = std::env::temp_dir().join(format!("qwanto-memory-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let mut memory = ProjectMemory::for_workspace(&root);
        memory.summary = "local fixture".into();
        memory.task_checkpoints.push("resume step".into());
        ProjectMemoryStore::save(&root, memory).unwrap();
        let loaded = ProjectMemoryStore::load(&root).unwrap();
        assert_eq!(loaded.summary, "local fixture");
        assert_eq!(loaded.task_checkpoints, vec!["resume step"]);
        let exported = ProjectMemoryStore::export(&root).unwrap();
        assert!(exported.contains("local fixture"));
        ProjectMemoryStore::clear(&root).unwrap();
        assert!(!ProjectMemoryStore::path(&root).exists());
        let _ = fs::remove_dir_all(root);
    }
}
