pub mod runtime_manager;
pub mod model_registry;
pub mod telemetry;
pub mod permission_policy;
pub mod tool_executor;
pub mod session_store;

use std::path::PathBuf;
use std::sync::Mutex;
use tauri::{AppHandle, State};
use runtime_manager::{QwantoRuntimeManager, RuntimeStatus, StartOptions};
use model_registry::{ModelInfo, ModelRegistry};
use telemetry::{TelemetryCollector, TelemetrySnapshot};
use permission_policy::{ExecutionMode, PermissionPolicy};
use tool_executor::{ToolExecutor, ToolResult};
use session_store::{AgentSession, SessionStore};

pub struct AppState {
    pub runtime_manager: Mutex<QwantoRuntimeManager>,
    pub permission_policy: Mutex<PermissionPolicy>,
    pub session_store: Mutex<SessionStore>,
}

#[tauri::command]
fn discover_models(directories: Vec<String>) -> Result<Vec<ModelInfo>, String> {
    Ok(ModelRegistry::discover_models(directories))
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
    state: State<AppState>,
    app: AppHandle,
) -> Result<(), String> {
    let manager = state.runtime_manager.lock().map_err(|e| e.to_string())?;
    manager.send_prompt(&request_id, &prompt, max_tokens, app)
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
fn set_workspace_root(root_path: String, state: State<AppState>) -> Result<(), String> {
    let path = PathBuf::from(&root_path);
    if !path.exists() || !path.is_dir() {
        return Err(format!("Workspace root does not exist: {}", root_path));
    }
    let mut policy = state.permission_policy.lock().map_err(|e| e.to_string())?;
    policy.workspace_root = Some(path);
    Ok(())
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
    tool_name: String,
    args: serde_json::Value,
    approved: bool,
    state: State<AppState>,
) -> Result<ToolResult, String> {
    let policy = state.permission_policy.lock().map_err(|e| e.to_string())?;

    match tool_name.as_str() {
        "read_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).ok_or("Missing 'path' argument")?;
            Ok(ToolExecutor::read_file(path, &policy))
        }
        "write_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).ok_or("Missing 'path' argument")?;
            let content = args.get("content").and_then(|v| v.as_str()).ok_or("Missing 'content' argument")?;
            Ok(ToolExecutor::write_file(path, content, &policy, approved))
        }
        "edit_file" => {
            let path = args.get("path").and_then(|v| v.as_str()).ok_or("Missing 'path' argument")?;
            let old_str = args.get("old_str").and_then(|v| v.as_str()).ok_or("Missing 'old_str' argument")?;
            let new_str = args.get("new_str").and_then(|v| v.as_str()).ok_or("Missing 'new_str' argument")?;
            Ok(ToolExecutor::edit_file(path, old_str, new_str, &policy, approved))
        }
        "list_directory" => {
            let path = args.get("path").and_then(|v| v.as_str());
            Ok(ToolExecutor::list_directory(path, &policy))
        }
        "execute_command" => {
            let cmd = args.get("command").and_then(|v| v.as_str()).ok_or("Missing 'command' argument")?;
            let cwd = args.get("cwd").and_then(|v| v.as_str());
            Ok(ToolExecutor::execute_command(cmd, cwd, &policy, approved))
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState {
            runtime_manager: Mutex::new(QwantoRuntimeManager::new()),
            permission_policy: Mutex::new(PermissionPolicy::new(None, ExecutionMode::Plan)),
            session_store: Mutex::new(SessionStore::new(None)),
        })
        .invoke_handler(tauri::generate_handler![
            discover_models,
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
            get_agent_session
        ])
        .run(tauri::generate_context!())
        .expect("failed to run the Qwanto desktop application");
}
