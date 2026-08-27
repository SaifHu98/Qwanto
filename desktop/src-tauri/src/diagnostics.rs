use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::permission_policy::PermissionPolicy;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct FeedbackBundle {
    pub relative_path: String,
    pub category: String,
    pub includes_logs: bool,
    pub includes_screenshot: bool,
}

pub fn record_error(root: &Path, source: &str, message: &str, context: &str) -> Result<(), String> {
    let source = source.trim();
    let message = message.trim();
    if source.is_empty() || source.len() > 128 || message.is_empty() || message.len() > 20_000 {
        return Err("Diagnostic error record is invalid.".into());
    }
    let root = root.canonicalize().map_err(|error| format!("Diagnostic root is unavailable: {error}"))?;
    let directory = root.join(".qwanto").join("diagnostics");
    fs::create_dir_all(&directory).map_err(|error| format!("Could not create diagnostics storage: {error}"))?;
    let directory = directory.canonicalize().map_err(|error| format!("Diagnostics storage is unavailable: {error}"))?;
    if !directory.starts_with(root.as_path()) { return Err("Diagnostics storage escaped its root boundary.".into()); }
    let root_text = root.to_string_lossy();
    let redact = |value: &str| PermissionPolicy::redact_secrets(value).replace(root_text.as_ref(), "[REDACTED_ROOT]");
    let record = serde_json::json!({
        "created_at": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        "source": source,
        "message": redact(message),
        "context": redact(context),
    });
    let path = directory.join("errors.jsonl");
    if path.metadata().map(|meta| meta.len() > 1_048_576).unwrap_or(false) {
        let rotated = directory.join("errors.jsonl.1");
        let _ = fs::remove_file(&rotated);
        fs::rename(&path, &rotated).map_err(|error| format!("Could not rotate diagnostics log: {error}"))?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(&path)
        .map_err(|error| format!("Could not open diagnostics log: {error}"))?;
    writeln!(file, "{}", serde_json::to_string(&record).map_err(|error| error.to_string())?)
        .map_err(|error| format!("Could not write diagnostics log: {error}"))?;
    file.sync_data().map_err(|error| format!("Could not flush diagnostics log: {error}"))
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffff;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 { (crc >> 1) ^ 0xedb8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

fn push_u16(output: &mut Vec<u8>, value: u16) { output.extend_from_slice(&value.to_le_bytes()); }
fn push_u32(output: &mut Vec<u8>, value: u32) { output.extend_from_slice(&value.to_le_bytes()); }

fn zip_entry(output: &mut Vec<u8>, central: &mut Vec<u8>, name: &str, data: &[u8]) {
    let name_bytes = name.as_bytes();
    let crc = crc32(data);
    let offset = output.len() as u32;
    push_u32(output, 0x0403_4b50);
    push_u16(output, 20); push_u16(output, 0); push_u16(output, 0); push_u16(output, 0);
    push_u16(output, 0); push_u16(output, 0); push_u32(output, crc);
    push_u32(output, data.len() as u32); push_u32(output, data.len() as u32);
    push_u16(output, name_bytes.len() as u16); push_u16(output, 0);
    output.extend_from_slice(name_bytes); output.extend_from_slice(data);

    push_u32(central, 0x0201_4b50);
    push_u16(central, 20); push_u16(central, 20); push_u16(central, 0); push_u16(central, 0);
    push_u16(central, 0); push_u16(central, 0); push_u32(central, crc);
    push_u32(central, data.len() as u32); push_u32(central, data.len() as u32);
    push_u16(central, name_bytes.len() as u16); push_u16(central, 0); push_u16(central, 0);
    push_u16(central, 0); push_u16(central, 0); push_u32(central, 0); push_u32(central, offset);
    central.extend_from_slice(name_bytes);
}

pub fn create(
    root: &Path,
    category: &str,
    description: &str,
    logs: &str,
    screenshot: Option<&[u8]>,
) -> Result<FeedbackBundle, String> {
    let category = category.trim();
    if category.is_empty() || category.len() > 64 { return Err("Feedback category is invalid.".into()); }
    if description.trim().is_empty() || description.len() > 20_000 { return Err("Feedback description is required and must be under 20,000 characters.".into()); }
    let workspace = root.canonicalize().map_err(|error| format!("Workspace is unavailable: {error}"))?;
    let directory = workspace.join(".qwanto").join("diagnostics");
    fs::create_dir_all(&directory).map_err(|error| format!("Could not create diagnostics storage: {error}"))?;
    let directory = directory.canonicalize().map_err(|error| format!("Diagnostics storage is unavailable: {error}"))?;
    if !directory.starts_with(workspace.as_path()) { return Err("Diagnostics storage escaped the workspace boundary.".into()); }

    let workspace_text = workspace.to_string_lossy().into_owned();
    let redacted_description = PermissionPolicy::redact_secrets(description).replace(&workspace_text, "[REDACTED_WORKSPACE]");
    let redacted_logs = PermissionPolicy::redact_secrets(logs).replace(&workspace_text, "[REDACTED_WORKSPACE]");
    let metadata = serde_json::json!({
        "product": "Qwanto Native",
        "surface": "Qwanto Code",
        "app_version": env!("CARGO_PKG_VERSION"),
        "os": std::env::consts::OS,
        "category": category,
        "description": redacted_description,
        "includes_logs": !logs.trim().is_empty(),
        "includes_screenshot": screenshot.is_some(),
        "created_at": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
    });
    let mut archive = Vec::new();
    let mut central = Vec::new();
    zip_entry(&mut archive, &mut central, "feedback.json", serde_json::to_string_pretty(&metadata).unwrap_or_default().as_bytes());
    if !redacted_logs.trim().is_empty() { zip_entry(&mut archive, &mut central, "redacted-logs.txt", redacted_logs.as_bytes()); }
    if let Some(image) = screenshot { zip_entry(&mut archive, &mut central, "screenshot.bin", image); }
    let central_offset = archive.len() as u32;
    archive.extend_from_slice(&central);
    push_u32(&mut archive, 0x0605_4b50); push_u16(&mut archive, 0); push_u16(&mut archive, 0);
    push_u16(&mut archive, 2); push_u16(&mut archive, 2); push_u32(&mut archive, central.len() as u32);
    push_u32(&mut archive, central_offset); push_u16(&mut archive, 0);

    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
    let path = directory.join(format!("feedback-{stamp}-{}.zip", std::process::id()));
    let mut file = fs::File::create(&path).map_err(|error| error.to_string())?;
    file.write_all(&archive).map_err(|error| error.to_string())?;
    file.sync_all().map_err(|error| error.to_string())?;
    Ok(FeedbackBundle { relative_path: path.strip_prefix(&workspace).map_err(|error| error.to_string())?.to_string_lossy().replace('\\', "/"), category: category.to_string(), includes_logs: !logs.trim().is_empty(), includes_screenshot: screenshot.is_some() })
}
