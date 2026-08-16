pub mod runtime_manager;
pub mod model_registry;
pub mod telemetry;

use std::sync::Mutex;
use tauri::{AppHandle, State};
use runtime_manager::{QwantoRuntimeManager, RuntimeStatus, StartOptions};
use model_registry::{ModelInfo, ModelRegistry};
use telemetry::{TelemetryCollector, TelemetrySnapshot};

pub struct AppState {
    pub runtime_manager: Mutex<QwantoRuntimeManager>,
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AppState {
            runtime_manager: Mutex::new(QwantoRuntimeManager::new()),
        })
        .invoke_handler(tauri::generate_handler![
            discover_models,
            start_model,
            stop_model,
            send_prompt,
            cancel_generation,
            get_runtime_status,
            get_telemetry_snapshot
        ])
        .run(tauri::generate_context!())
        .expect("failed to run the Qwanto desktop application");
}
