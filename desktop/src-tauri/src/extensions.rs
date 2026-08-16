use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};

pub const ALLOWED_CAPABILITIES: &[&str] = &[
    "workspace.read", "workspace.write", "terminal.execute", "git.read", "git.write",
    "github.read", "github.write", "network.search", "model.control", "diagnostics.read",
    "secrets.access",
];

const DANGEROUS_CAPABILITIES: &[&str] = &[
    "workspace.write", "terminal.execute", "git.write", "github.write", "network.search",
    "model.control", "secrets.access",
];

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PluginPublisher {
    pub name: String,
    pub key_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PluginManifest {
    pub schema_version: u8,
    pub id: String,
    pub name: String,
    pub publisher: PluginPublisher,
    pub version: String,
    pub sha256: String,
    pub requested_capabilities: Vec<String>,
    pub entrypoint: String,
    pub signature: String,
    #[serde(default)]
    pub source_url: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PluginValidation {
    pub valid: bool,
    pub errors: Vec<String>,
    pub package_sha256: String,
    pub dangerous_capabilities: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InstalledPlugin {
    pub manifest: PluginManifest,
    pub enabled: bool,
    pub quarantined: bool,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.chars().all(|character| character.is_ascii_hexdigit())
}

fn is_identifier(value: &str) -> bool {
    !value.is_empty() && value.len() <= 81 && value.chars().next().is_some_and(|character| character.is_ascii_lowercase() || character.is_ascii_digit()) && value.chars().enumerate().all(|(index, character)| {
        character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-' || character == '_' || (index > 0 && character == '.')
    })
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

pub fn validate_manifest(manifest: &PluginManifest, package_bytes: &[u8]) -> PluginValidation {
    let package_sha256 = sha256_hex(package_bytes);
    let mut errors = Vec::new();
    if manifest.schema_version != 1 { errors.push("schema_version must be 1.".into()); }
    if !is_identifier(&manifest.id) { errors.push("id must be a short lowercase package identifier.".into()); }
    if manifest.name.trim().is_empty() { errors.push("name is required.".into()); }
    if manifest.publisher.name.trim().is_empty() || manifest.publisher.key_id.trim().is_empty() { errors.push("publisher.name and publisher.key_id are required.".into()); }
    if manifest.version.trim().is_empty() { errors.push("version is required.".into()); }
    if !is_sha256(&manifest.sha256) { errors.push("sha256 must be a 64-character hexadecimal digest.".into()); }
    if manifest.sha256.to_ascii_lowercase() != package_sha256 { errors.push("sha256 does not match the package bytes.".into()); }
    if manifest.requested_capabilities.is_empty() { errors.push("requested_capabilities must contain at least one capability.".into()); }
    for capability in &manifest.requested_capabilities {
        if !ALLOWED_CAPABILITIES.contains(&capability.as_str()) { errors.push(format!("Unsupported capability: {capability}.")); }
    }
    if manifest.entrypoint.trim().is_empty() || Path::new(&manifest.entrypoint).is_absolute() || manifest.entrypoint.split(['/', '\\']).any(|part| part == "..") {
        errors.push("entrypoint must be a package-relative path.".into());
    }
    if manifest.signature.trim().is_empty() { errors.push("signature is required; unsigned plugins cannot be enabled.".into()); }
    let dangerous_capabilities = manifest.requested_capabilities.iter().filter(|capability| DANGEROUS_CAPABILITIES.contains(&capability.as_str())).cloned().collect();
    PluginValidation { valid: errors.is_empty(), errors, package_sha256, dangerous_capabilities }
}

fn registry_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app.path().app_data_dir().map_err(|error| error.to_string())?.join("extensions"))
}

fn registry_path(app: &AppHandle) -> Result<PathBuf, String> { Ok(registry_dir(app)?.join("plugins.json")) }

fn read_registry(app: &AppHandle) -> Result<Vec<InstalledPlugin>, String> {
    let path = registry_path(app)?;
    if !path.exists() { return Ok(Vec::new()); }
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    serde_json::from_slice(&bytes).map_err(|error| error.to_string())
}

fn write_registry(app: &AppHandle, plugins: &[InstalledPlugin]) -> Result<(), String> {
    let directory = registry_dir(app)?;
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let path = registry_path(app)?;
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(plugins).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
    fs::rename(temporary, path).map_err(|error| error.to_string())
}

pub fn list_plugins(app: &AppHandle) -> Result<Vec<InstalledPlugin>, String> { read_registry(app) }

pub fn install_plugin(app: &AppHandle, manifest: PluginManifest, package: &[u8]) -> Result<PluginValidation, String> {
    let validation = validate_manifest(&manifest, package);
    if !validation.valid { return Ok(validation); }
    let directory = registry_dir(app)?.join(&manifest.id);
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    fs::write(directory.join("plugin.json"), serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?).map_err(|error| error.to_string())?;
    fs::write(directory.join("package.bin"), package).map_err(|error| error.to_string())?;
    let mut plugins = read_registry(app)?.into_iter().filter(|plugin| plugin.manifest.id != manifest.id).collect::<Vec<_>>();
    plugins.push(InstalledPlugin { manifest, enabled: false, quarantined: false });
    write_registry(app, &plugins)?;
    Ok(validation)
}

pub fn set_plugin_enabled(app: &AppHandle, id: &str, enabled: bool) -> Result<(), String> {
    let mut plugins = read_registry(app)?;
    let plugin = plugins.iter_mut().find(|plugin| plugin.manifest.id == id).ok_or_else(|| "Plugin is not installed.".to_string())?;
    if enabled && plugin.quarantined { return Err("Quarantined plugins must be restored and reviewed before enabling.".into()); }
    plugin.enabled = enabled;
    write_registry(app, &plugins)
}

pub fn quarantine_plugin(app: &AppHandle, id: &str) -> Result<(), String> {
    let mut plugins = read_registry(app)?;
    let plugin = plugins.iter_mut().find(|plugin| plugin.manifest.id == id).ok_or_else(|| "Plugin is not installed.".to_string())?;
    plugin.enabled = false;
    plugin.quarantined = true;
    write_registry(app, &plugins)
}

pub fn uninstall_plugin(app: &AppHandle, id: &str) -> Result<(), String> {
    let mut plugins = read_registry(app)?;
    let removed = plugins.iter().any(|plugin| plugin.manifest.id == id);
    if !removed { return Err("Plugin is not installed.".into()); }
    plugins.retain(|plugin| plugin.manifest.id != id);
    let directory = registry_dir(app)?.join(id);
    if directory.exists() { fs::remove_dir_all(directory).map_err(|error| error.to_string())?; }
    write_registry(app, &plugins)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(bytes: &[u8]) -> PluginManifest {
        PluginManifest {
            schema_version: 1,
            id: "local-reviewer".into(),
            name: "Local Reviewer".into(),
            publisher: PluginPublisher { name: "Qwanto Test Publisher".into(), key_id: "test-key".into() },
            version: "1.0.0".into(),
            sha256: sha256_hex(bytes),
            requested_capabilities: vec!["workspace.read".into(), "git.read".into()],
            entrypoint: "bin/reviewer".into(),
            signature: "test-signature".into(),
            source_url: None,
            license: Some("MIT".into()),
        }
    }

    #[test]
    fn rejects_checksum_mismatch_and_unknown_capability() {
        let bytes = b"package";
        let mut candidate = manifest(bytes);
        candidate.sha256 = "0".repeat(64);
        candidate.requested_capabilities.push("shell.root".into());
        let result = validate_manifest(&candidate, bytes);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|error| error.contains("sha256 does not match")));
        assert!(result.errors.iter().any(|error| error.contains("Unsupported capability")));
    }

    #[test]
    fn dangerous_capabilities_are_reported_for_approval() {
        let bytes = b"package";
        let mut candidate = manifest(bytes);
        candidate.requested_capabilities.push("github.write".into());
        let result = validate_manifest(&candidate, bytes);
        assert!(result.valid);
        assert_eq!(result.dangerous_capabilities, vec!["github.write"]);
    }
}
