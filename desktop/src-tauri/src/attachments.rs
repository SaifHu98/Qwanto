use std::fs;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

pub const MAX_ATTACHMENT_BYTES: usize = 10 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct StoredAttachment {
    pub id: String,
    pub name: String,
    pub mime: String,
    pub size: usize,
    pub relative_path: String,
    pub previewable: bool,
}

fn safe_name(input: &str) -> Result<String, String> {
    let name = Path::new(input)
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| "Attachment name is invalid.".to_string())?;
    let mut output = String::with_capacity(name.len().min(128));
    for character in name.chars().take(128) {
        if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_' | ' ') {
            output.push(character);
        } else {
            output.push('_');
        }
    }
    let output = output.trim().trim_matches('.').to_string();
    if output.is_empty() || output == "." || output == ".." {
        return Err("Attachment name is invalid.".into());
    }
    Ok(output)
}

pub fn store(root: &Path, name: &str, mime: &str, bytes: &[u8]) -> Result<StoredAttachment, String> {
    if bytes.is_empty() {
        return Err("Attachment is empty.".into());
    }
    if bytes.len() > MAX_ATTACHMENT_BYTES {
        return Err(format!("Attachments are limited to {} MiB.", MAX_ATTACHMENT_BYTES / (1024 * 1024)));
    }
    let name = safe_name(name)?;
    let mime = if mime.trim().is_empty() { "application/octet-stream" } else { mime.trim() };
    if mime.len() > 128 || mime.chars().any(char::is_whitespace) {
        return Err("Attachment MIME type is invalid.".into());
    }

    let workspace = root.canonicalize().map_err(|error| format!("Workspace is unavailable: {error}"))?;
    let directory = workspace.join(".qwanto").join("attachments");
    fs::create_dir_all(&directory).map_err(|error| format!("Could not create attachment storage: {error}"))?;
    let canonical_directory = directory
        .canonicalize()
        .map_err(|error| format!("Attachment storage is unavailable: {error}"))?;
    if !canonical_directory.starts_with(&workspace) {
        return Err("Attachment storage escaped the workspace boundary.".into());
    }

    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    let id = format!("attachment-{}-{stamp}", std::process::id());
    let path = canonical_directory.join(format!("{id}-{name}"));
    if path.exists() {
        return Err("Attachment storage collision; please retry.".into());
    }
    fs::write(&path, bytes).map_err(|error| format!("Could not store attachment: {error}"))?;

    Ok(StoredAttachment {
        id,
        name,
        mime: mime.to_string(),
        size: bytes.len(),
        relative_path: path
            .strip_prefix(&workspace)
            .map_err(|error| error.to_string())?
            .to_string_lossy()
            .replace('\\', "/"),
        previewable: mime.starts_with("image/") || mime.starts_with("text/"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_only_inside_workspace_and_sanitizes_name() {
        let root = std::env::temp_dir().join(format!("qwanto-attachments-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        let stored = store(&root, "../notes:one.txt", "text/plain", b"local").unwrap();
        assert!(stored.relative_path.starts_with(".qwanto/attachments/"));
        assert!(root.join(&stored.relative_path).is_file());
        assert!(!stored.name.contains(".."));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rejects_oversized_attachment() {
        let root = std::env::temp_dir();
        let bytes = vec![0_u8; MAX_ATTACHMENT_BYTES + 1];
        assert!(store(&root, "large.bin", "application/octet-stream", &bytes).is_err());
    }
}
