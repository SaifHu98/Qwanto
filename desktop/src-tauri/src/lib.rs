pub mod runtime_manager;
pub mod gateway_manager;
pub mod model_registry;
pub mod telemetry;
pub mod permission_policy;
pub mod tool_executor;
pub mod session_store;
pub mod project_memory;
pub mod attachments;
pub mod diagnostics;
pub mod extensions;

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::fs;
use std::sync::Mutex;
use tauri::{AppHandle, Manager, RunEvent, State};
use tauri_plugin_dialog::{DialogExt, FilePath};
use gateway_manager::{GatewayManager, GatewayStatus};
use runtime_manager::{QwantoRuntimeManager, RuntimeStatus, StartOptions};
use model_registry::{ModelInfo, ModelRegistry};
use telemetry::{TelemetryCollector, TelemetrySnapshot};
use permission_policy::{ExecutionMode, PermissionPolicy};
use tool_executor::{ToolExecutor, ToolResult};
use session_store::{AgentSession, SessionStore};
use project_memory::{ProjectMemory, ProjectMemoryStore};
use attachments::StoredAttachment;
use diagnostics::FeedbackBundle;
use extensions::{InstalledPlugin, PluginManifest, PluginValidation};

pub struct AppState {
    pub gateway_manager: Mutex<GatewayManager>,
    pub runtime_manager: Mutex<QwantoRuntimeManager>,
    pub permission_policy: Mutex<PermissionPolicy>,
    pub session_store: Mutex<SessionStore>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct DesktopCapabilities {
    converter_available: bool,
    downloader_available: bool,
    gateway_sidecar_packaged: bool,
    reason: String,
}

#[tauri::command]
fn get_desktop_capabilities(state: State<AppState>) -> DesktopCapabilities {
    let gateway_sidecar_packaged = state
        .gateway_manager
        .lock()
        .map(|mut manager| manager.status().sidecar_packaged)
        .unwrap_or(false);
    DesktopCapabilities {
        converter_available: true,
        downloader_available: true,
        gateway_sidecar_packaged,
        reason: if gateway_sidecar_packaged {
            "The local gateway sidecar provides model discovery, conversion, and consent-gated acquisition.".into()
        } else {
            "Development mode uses the repository gateway; release packages include the target-native sidecar.".into()
        },
    }
}

#[tauri::command]
fn get_gateway_status(state: State<AppState>) -> Result<GatewayStatus, String> {
    let mut manager = state.gateway_manager.lock().map_err(|error| error.to_string())?;
    Ok(manager.status())
}

#[tauri::command]
fn restart_gateway(state: State<AppState>) -> Result<GatewayStatus, String> {
    let mut manager = state.gateway_manager.lock().map_err(|error| error.to_string())?;
    manager.restart()
}

#[tauri::command]
fn discover_models(directories: Vec<String>) -> Result<Vec<ModelInfo>, String> {
    Ok(ModelRegistry::discover_models(directories))
}

fn selected_path(path: FilePath) -> Result<String, String> {
    match path {
        FilePath::Path(path) => Ok(path.to_string_lossy().to_string()),
        FilePath::Url(url) => Err(format!("The selected location is not a local filesystem path: {url}")),
    }
}

#[tauri::command]
fn pick_model_source(app: AppHandle) -> Result<Option<String>, String> {
    app.dialog().file()
        .add_filter("Qwanto model sources", &["qwn", "gguf", "safetensors", "pt", "pth", "bin"])
        .blocking_pick_file()
        .map(selected_path)
        .transpose()
}

#[tauri::command]
fn pick_qwn_model(app: AppHandle) -> Result<Option<String>, String> {
    app.dialog().file()
        .add_filter("Qwanto Native containers", &["qwn"])
        .blocking_pick_file()
        .map(selected_path)
        .transpose()
}

#[tauri::command]
fn pick_model_library_folder(app: AppHandle) -> Result<Option<String>, String> {
    app.dialog().file().blocking_pick_folder()
        .map(selected_path)
        .transpose()
}

#[tauri::command]
fn pick_workspace_folder(app: AppHandle) -> Result<Option<String>, String> {
    app.dialog().file().blocking_pick_folder()
        .map(selected_path)
        .transpose()
}

#[tauri::command]
fn start_model(
    model_path: String,
    options: Option<StartOptions>,
    state: State<AppState>,
    app: AppHandle,
) -> Result<RuntimeStatus, String> {
    let manager = state.runtime_manager.lock().map_err(|e| e.to_string())?;
    manager.start_model(&model_path, options, app)
}

#[tauri::command]
fn stop_model(state: State<AppState>) -> Result<RuntimeStatus, String> {
    let manager = state.runtime_manager.lock().map_err(|e| e.to_string())?;
    manager.stop_model()
}

#[tauri::command]
fn send_prompt(
    request_id: String,
    prompt: String,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    state: State<AppState>,
) -> Result<(), String> {
    let manager = state.runtime_manager.lock().map_err(|e| e.to_string())?;
    manager.send_prompt(&request_id, &prompt, max_tokens, temperature, top_p)
}

#[tauri::command]
fn cancel_generation(request_id: String, state: State<AppState>) -> Result<(), String> {
    let manager = state.runtime_manager.lock().map_err(|e| e.to_string())?;
    manager.cancel_generation(&request_id)
}

#[tauri::command]
fn get_runtime_status(state: State<AppState>) -> Result<RuntimeStatus, String> {
    let manager = state.runtime_manager.lock().map_err(|e| e.to_string())?;
    Ok(manager.get_status())
}

#[tauri::command]
fn get_telemetry_snapshot() -> Result<TelemetrySnapshot, String> {
    Ok(TelemetryCollector::get_snapshot(None, None, 0))
}

#[tauri::command]
fn set_workspace_root(root_path: String, state: State<AppState>) -> Result<String, String> {
    let path = PathBuf::from(&root_path);
    let mut policy = state.permission_policy.lock().map_err(|e| e.to_string())?;
    let canonical = policy.set_workspace_root(&path)?;
    Ok(canonical.to_string_lossy().to_string())
}

#[tauri::command]
fn set_execution_mode(mode: String, state: State<AppState>) -> Result<(), String> {
    let mut policy = state.permission_policy.lock().map_err(|e| e.to_string())?;
    policy.mode = match mode.to_lowercase().as_str() {
        "plan" => ExecutionMode::Plan,
        _ => ExecutionMode::Agent,
    };
    Ok(())
}

#[tauri::command]
fn execute_agent_tool(
    session_id: String,
    tool_name: String,
    args: serde_json::Value,
    approval_token: Option<String>,
    state: State<AppState>,
) -> Result<ToolResult, String> {
    let policy = state.permission_policy.lock().map_err(|e| e.to_string())?;

    match tool_name.as_str() {
        "read_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).ok_or("Missing 'path' argument")?;
            Ok(ToolExecutor::read_file(&session_id, path, &policy))
        }
        "write_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).ok_or("Missing 'path' argument")?;
            let content = args.get("content").and_then(|v| v.as_str()).ok_or("Missing 'content' argument")?;
            Ok(ToolExecutor::write_file(&session_id, path, content, approval_token.as_deref(), &policy))
        }
        "edit_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).ok_or("Missing 'path' argument")?;
            let old_str = args.get("old_str").and_then(|v| v.as_str()).ok_or("Missing 'old_str' argument")?;
            let new_str = args.get("new_str").and_then(|v| v.as_str()).ok_or("Missing 'new_str' argument")?;
            Ok(ToolExecutor::edit_file(&session_id, path, old_str, new_str, approval_token.as_deref(), &policy))
        }
        "list_directory" => {
            let path = args.get("path").and_then(|v| v.as_str());
            Ok(ToolExecutor::list_directory(&session_id, path, &policy))
        }
        "execute_command" => {
            let program = args.get("program").and_then(|v| v.as_str()).unwrap_or("powershell");
            let cmd_args: Vec<String> = args.get("args")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let cwd = args.get("cwd").and_then(|v| v.as_str());
            Ok(ToolExecutor::execute_command(&session_id, program, cmd_args, cwd, approval_token.as_deref(), &policy))
        }
        _ => Err(format!("Unsupported agent tool: {}", tool_name)),
    }
}

#[tauri::command]
fn list_agent_sessions(state: State<AppState>) -> Result<Vec<AgentSession>, String> {
    let store = state.session_store.lock().map_err(|e| e.to_string())?;
    Ok(store.list_sessions())
}

#[tauri::command]
fn save_agent_session(session: AgentSession, state: State<AppState>) -> Result<(), String> {
    let store = state.session_store.lock().map_err(|e| e.to_string())?;
    store.save_session(&session)
}

#[tauri::command]
fn get_agent_session(session_id: String, state: State<AppState>) -> Result<Option<AgentSession>, String> {
    let store = state.session_store.lock().map_err(|e| e.to_string())?;
    Ok(store.get_session(&session_id))
}

fn workspace_root(state: &State<AppState>) -> Result<PathBuf, String> {
    let policy = state.permission_policy.lock().map_err(|error| error.to_string())?;
    policy
        .workspace_root
        .clone()
        .ok_or_else(|| "Open a local project before using project memory.".to_string())
}

#[tauri::command]
fn get_project_memory(state: State<AppState>) -> Result<ProjectMemory, String> {
    let root = workspace_root(&state)?;
    ProjectMemoryStore::load(&root)
}

#[tauri::command]
fn save_project_memory(memory: ProjectMemory, state: State<AppState>) -> Result<ProjectMemory, String> {
    let root = workspace_root(&state)?;
    ProjectMemoryStore::save(&root, memory)
}

#[tauri::command]
fn clear_project_memory(state: State<AppState>) -> Result<ProjectMemory, String> {
    let root = workspace_root(&state)?;
    ProjectMemoryStore::clear(&root)
}

#[tauri::command]
fn export_project_memory(app: AppHandle, state: State<AppState>) -> Result<Option<String>, String> {
    let root = workspace_root(&state)?;
    let content = ProjectMemoryStore::export(&root)?;
    let Some(selected) = app.dialog().file()
        .set_file_name("qwanto-project-memory.json")
        .blocking_save_file() else {
        return Ok(None);
    };
    let path = match selected {
        FilePath::Path(path) => path,
        FilePath::Url(url) => return Err(format!("The selected location is not a local filesystem path: {url}")),
    };
    fs::write(&path, content).map_err(|error| format!("Could not export project memory: {error}"))?;
    Ok(Some(path.to_string_lossy().to_string()))
}

#[tauri::command]
fn set_project_memory_enabled(enabled: bool, state: State<AppState>) -> Result<ProjectMemory, String> {
    let root = workspace_root(&state)?;
    let mut memory = ProjectMemoryStore::load(&root)?;
    memory.enabled = enabled;
    ProjectMemoryStore::save(&root, memory)
}

#[tauri::command]
fn store_chat_attachment(name: String, mime: String, bytes: Vec<u8>, state: State<AppState>) -> Result<StoredAttachment, String> {
    let root = workspace_root(&state)?;
    attachments::store(&root, &name, &mime, &bytes)
}

fn attachment_mime(path: &std::path::Path) -> &'static str {
    match path.extension().and_then(|value| value.to_str()).unwrap_or_default().to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "svg" => "image/svg+xml",
        "txt" | "md" | "rs" | "c" | "h" | "cpp" | "py" | "js" | "ts" | "tsx" | "json" | "toml" | "yaml" | "yml" => "text/plain",
        _ => "application/octet-stream",
    }
}

#[tauri::command]
fn pick_chat_attachment(app: AppHandle, state: State<AppState>) -> Result<Option<StoredAttachment>, String> {
    let Some(selected) = app.dialog().file().blocking_pick_file() else { return Ok(None) };
    let path = match selected { FilePath::Path(path) => path, FilePath::Url(url) => return Err(format!("The selected location is not a local filesystem path: {url}")) };
    let metadata = fs::metadata(&path).map_err(|error| format!("Could not inspect attachment: {error}"))?;
    if metadata.len() == 0 || metadata.len() > attachments::MAX_ATTACHMENT_BYTES as u64 {
        return Err(format!("Attachments must be between 1 byte and {} MiB.", attachments::MAX_ATTACHMENT_BYTES / (1024 * 1024)));
    }
    let bytes = fs::read(&path).map_err(|error| format!("Could not read attachment: {error}"))?;
    let root = workspace_root(&state)?;
    attachments::store(&root, path.file_name().and_then(|value| value.to_str()).unwrap_or("attachment"), attachment_mime(&path), &bytes).map(Some)
}

#[derive(Debug, Clone, serde::Serialize)]
struct PickedFileBytes {
    name: String,
    bytes: Vec<u8>,
}

fn pick_file_bytes(
    app: &AppHandle,
    title: &str,
    extensions: &[&str],
    max_bytes: u64,
) -> Result<Option<PickedFileBytes>, String> {
    let Some(selected) = app.dialog().file().add_filter(title, extensions).blocking_pick_file() else {
        return Ok(None);
    };
    let path = match selected {
        FilePath::Path(path) => path,
        FilePath::Url(url) => return Err(format!("The selected location is not a local filesystem path: {url}")),
    };
    let metadata = fs::metadata(&path).map_err(|error| format!("Could not inspect selected file: {error}"))?;
    if metadata.len() == 0 || metadata.len() > max_bytes {
        return Err(format!("The selected file must be between 1 byte and {} MiB.", max_bytes / (1024 * 1024)));
    }
    let name = path.file_name().and_then(|value| value.to_str()).unwrap_or("selected-file").to_string();
    let bytes = fs::read(&path).map_err(|error| format!("Could not read selected file: {error}"))?;
    Ok(Some(PickedFileBytes { name, bytes }))
}

#[tauri::command]
fn pick_plugin_package(app: AppHandle) -> Result<Option<PickedFileBytes>, String> {
    pick_file_bytes(&app, "Qwanto plugin packages", &["zip", "qwp", "tar", "gz", "bin"], 25 * 1024 * 1024)
}

#[tauri::command]
fn pick_feedback_screenshot(app: AppHandle) -> Result<Option<PickedFileBytes>, String> {
    pick_file_bytes(&app, "Feedback screenshots", &["png", "jpg", "jpeg", "webp"], 5 * 1024 * 1024)
}

#[tauri::command]
fn create_feedback_bundle(category: String, description: String, logs: String, screenshot: Option<Vec<u8>>, state: State<AppState>) -> Result<FeedbackBundle, String> {
    let root = workspace_root(&state)?;
    diagnostics::create(&root, &category, &description, &logs, screenshot.as_deref())
}

#[tauri::command]
fn list_plugins(app: AppHandle) -> Result<Vec<InstalledPlugin>, String> { extensions::list_plugins(&app) }

#[tauri::command]
fn validate_plugin_manifest(manifest: PluginManifest, package: Vec<u8>) -> Result<PluginValidation, String> { Ok(extensions::validate_manifest(&manifest, &package)) }

#[tauri::command]
fn install_plugin(manifest: PluginManifest, package: Vec<u8>, app: AppHandle) -> Result<PluginValidation, String> { extensions::install_plugin(&app, manifest, &package) }

#[tauri::command]
fn set_plugin_enabled(id: String, enabled: bool, app: AppHandle) -> Result<(), String> { extensions::set_plugin_enabled(&app, &id, enabled) }

#[tauri::command]
fn quarantine_plugin(id: String, app: AppHandle) -> Result<(), String> { extensions::quarantine_plugin(&app, &id) }

#[tauri::command]
fn uninstall_plugin(id: String, app: AppHandle) -> Result<(), String> { extensions::uninstall_plugin(&app, &id) }

fn loopback_post_json(api_url: &str, path: &str, payload: serde_json::Value, desktop_search_token: Option<&str>) -> Result<serde_json::Value, String> {
    let authority = api_url
        .strip_prefix("http://")
        .ok_or_else(|| "Gateway search requires an HTTP loopback sidecar.".to_string())?
        .trim_end_matches('/');
    let mut parts = authority.splitn(2, ':');
    let host = parts.next().unwrap_or_default();
    if host != "127.0.0.1" && host != "localhost" && host != "[::1]" {
        return Err("Gateway search is restricted to the loopback sidecar.".into());
    }
    let port = parts
        .next()
        .ok_or_else(|| "Gateway readiness did not include a port.".to_string())?
        .parse::<u16>()
        .map_err(|error| format!("Invalid gateway port: {error}"))?;
    let body = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
    let address = (host.trim_matches(&['[', ']'][..]), port)
        .to_socket_addrs()
        .map_err(|error| format!("Gateway address unavailable: {error}"))?
        .next()
        .ok_or_else(|| "Gateway address unavailable.".to_string())?;
    let mut stream = TcpStream::connect_timeout(&address, std::time::Duration::from_secs(3))
        .map_err(|error| format!("Gateway search connection failed: {error}"))?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(15)))
        .map_err(|error| error.to_string())?;
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{}\r\n",
        body.len(),
        desktop_search_token.map(|token| format!("X-Qwanto-Desktop-Approval: {token}\r\n")).unwrap_or_default()
    );
    stream.write_all(request.as_bytes()).map_err(|error| error.to_string())?;
    stream.write_all(&body).map_err(|error| error.to_string())?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).map_err(|error| error.to_string())?;
    let separator = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| "Gateway returned an invalid HTTP response.".to_string())?;
    let header = String::from_utf8_lossy(&response[..separator]);
    if !header.starts_with("HTTP/1.1 200") && !header.starts_with("HTTP/1.0 200") {
        return Err(format!("Gateway search was rejected: {}", header.lines().next().unwrap_or("unknown status")));
    }
    serde_json::from_slice(&response[separator + 4..]).map_err(|error| format!("Invalid gateway search response: {error}"))
}

#[tauri::command]
fn web_search(query: String, approval_token: Option<String>, state: State<AppState>) -> Result<ToolResult, String> {
    let query = query.trim().to_string();
    if query.is_empty() || query.len() > 256 {
        return Err("Search query must be between 1 and 256 characters.".into());
    }
    let session_id = "desktop-internet";
    let args_hash = format!("web-search:{query}");
    let policy = state.permission_policy.lock().map_err(|error| error.to_string())?;
    let outcome = policy.evaluate_action(session_id, "web_search", None, Some("external web search"), &args_hash, Some(&query));
    let (token, details) = match outcome {
        permission_policy::PolicyOutcome::Deny { reason } => return Ok(ToolResult {
            success: false, outcome: "denied".into(), output: String::new(), error: Some(reason),
            truncated: false, approval_token: None, action_details: None,
        }),
        permission_policy::PolicyOutcome::NeedsApproval { token, details } => (token, details),
        permission_policy::PolicyOutcome::Allow => return Err("Web search must remain approval-gated.".into()),
    };
    let provided = match approval_token.as_deref() {
        Some(value) => value,
        None => return Ok(ToolResult {
            success: false, outcome: "needs_approval".into(), output: String::new(), error: None,
            truncated: false, approval_token: Some(token), action_details: Some(details),
        }),
    };
    let root = policy.workspace_root.clone().ok_or_else(|| "Open a local project before approving external search.".to_string())?;
    policy.token_registry.consume_token(provided, session_id, "web_search", &args_hash, &root, policy.mode)
        .map_err(|error| format!("Authorization Denied: {error}"))?;
    drop(policy);
    let (api_url, desktop_search_token) = {
        let mut manager = state.gateway_manager.lock().map_err(|error| error.to_string())?;
        let url = manager.status().api_url.ok_or_else(|| "Gateway is not ready.".to_string())?;
        let token = manager.desktop_search_token().ok_or_else(|| "Desktop search approval channel is unavailable.".to_string())?.to_string();
        (url, token)
    };
    let mut result = loopback_post_json(&api_url, "/v1/qwanto/search", serde_json::json!({"query": query}), Some(&desktop_search_token))?;
    if let Some(items) = result.get_mut("results").and_then(serde_json::Value::as_array_mut) {
        let timestamp = chrono_like_timestamp();
        for item in items {
            if let Some(object) = item.as_object_mut() {
                object.insert("timestamp".into(), serde_json::Value::String(timestamp.clone()));
                object.insert("included_in_context".into(), serde_json::Value::Bool(false));
            }
        }
    }
    Ok(ToolResult {
        success: true, outcome: "executed".into(), output: serde_json::to_string_pretty(&result).unwrap_or_default(),
        error: None, truncated: false, approval_token: None, action_details: None,
    })
}

fn chrono_like_timestamp() -> String {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs().to_string()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState {
            gateway_manager: Mutex::new(GatewayManager::new()),
            runtime_manager: Mutex::new(QwantoRuntimeManager::new()),
            permission_policy: Mutex::new(PermissionPolicy::new(None, ExecutionMode::Plan)),
            session_store: Mutex::new(SessionStore::new(None)),
        })
        .setup(|app| {
            let resource_dir = app.path().resource_dir().map_err(|error| error.to_string())?;
            let data_dir = app.path().app_data_dir().map_err(|error| error.to_string())?;
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                if let Ok(mut gateway) = handle.state::<AppState>().gateway_manager.lock() {
                    let _ = gateway.start(&resource_dir, &data_dir);
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            discover_models,
            pick_model_source,
            pick_qwn_model,
            pick_model_library_folder,
            pick_workspace_folder,
            get_desktop_capabilities,
            get_gateway_status,
            restart_gateway,
            start_model,
            stop_model,
            send_prompt,
            cancel_generation,
            get_runtime_status,
            get_telemetry_snapshot,
            set_workspace_root,
            set_execution_mode,
            execute_agent_tool,
            list_agent_sessions,
            save_agent_session,
            get_agent_session,
            get_project_memory,
            save_project_memory,
            clear_project_memory,
            export_project_memory,
            set_project_memory_enabled,
            store_chat_attachment,
            pick_chat_attachment,
            pick_plugin_package,
            pick_feedback_screenshot,
            create_feedback_bundle,
            list_plugins,
            validate_plugin_manifest,
            install_plugin,
            set_plugin_enabled,
            quarantine_plugin,
            uninstall_plugin,
            web_search
        ])
        .build(tauri::generate_context!())
        .expect("failed to build the Qwanto desktop application")
        .run(|app: &AppHandle, event| {
            if matches!(event, RunEvent::Exit | RunEvent::ExitRequested { .. }) {
                if let Ok(state) = app.state::<AppState>().gateway_manager.lock() {
                    let mut gateway = state;
                    gateway.stop();
                }
                if let Ok(runtime) = app.state::<AppState>().runtime_manager.lock() {
                    let _ = runtime.stop_model();
                }
            }
        })
}
