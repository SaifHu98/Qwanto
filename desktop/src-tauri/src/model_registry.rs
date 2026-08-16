use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub path: String,
    pub path_alias: String,
    pub size_formatted: String,
    pub size_bytes: u64,
    pub format: String,
    pub quantization: String,
    pub compatibility_state: String,
    pub metadata_status: String,
    pub n_tensors: Option<u32>,
    pub n_layers: Option<u32>,
}

pub struct ModelRegistry;

impl ModelRegistry {
    pub const QWN_MAGIC_QWN2: u32 = 0x51574E32; // "QWN2"
    pub const QWN_MAGIC_COLI: u32 = 0x434F4C49; // "COLI"
    pub const QWN_MAGIC_QWN1: u32 = 0x51574E31; // "QWN1"

    pub fn discover_models(directories: Vec<String>) -> Vec<ModelInfo> {
        let mut results = Vec::new();

        let candidate_dirs: Vec<PathBuf> = if !directories.is_empty() {
            directories.into_iter().map(PathBuf::from).collect()
        } else {
            vec![
                PathBuf::from("models"),
                PathBuf::from("../models"),
                PathBuf::from("experiments/results"),
                PathBuf::from("../experiments/results"),
            ]
        };

        for dir in candidate_dirs {
            if !dir.exists() || !dir.is_dir() {
                continue;
            }

            let canonical_dir = match dir.canonicalize() {
                Ok(d) => d,
                Err(_) => continue,
            };

            if let Ok(entries) = fs::read_dir(&canonical_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                        // Primary default acceptance: .qwn containers only
                        if name.ends_with(".qwn") {
                            let metadata = Self::inspect_qwn_file(&path);
                            results.push(metadata);
                        } else if name.ends_with(".gguf") {
                            // Isolated external runtime format - labeled unavailable by default
                            let metadata = Self::inspect_gguf_file(&path);
                            results.push(metadata);
                        }
                    }
                }
            }
        }

        results
    }

    pub fn inspect_qwn_file(path: &Path) -> ModelInfo {
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let size_bytes = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let size_gb = (size_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
        let size_formatted = format!("{:.2} GB", size_gb);
        let path_alias = format!("models://{}", name);

        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                return ModelInfo {
                    id: name.clone(),
                    name,
                    path: path.to_string_lossy().to_string(),
                    path_alias,
                    size_formatted,
                    size_bytes,
                    format: ".qwn container".into(),
                    quantization: "Unknown".into(),
                    compatibility_state: "unknown".into(),
                    metadata_status: format!("Unknown / validation unavailable: {}", e),
                    n_tensors: None,
                    n_layers: None,
                };
            }
        };

        // Read 4 KiB header
        let mut header_buf = [0u8; 4096];
        if let Err(e) = file.read_exact(&mut header_buf) {
            return ModelInfo {
                id: name.clone(),
                name,
                path: path.to_string_lossy().to_string(),
                path_alias,
                size_formatted,
                size_bytes,
                format: ".qwn container".into(),
                quantization: "Unknown".into(),
                compatibility_state: "invalid".into(),
                metadata_status: format!("Invalid: file smaller than 4KiB header ({})", e),
                n_tensors: None,
                n_layers: None,
            };
        }

        // Header layout: magic(4), version(4), n_tensors(4), n_layers(4), total_payload_bytes(8)
        let magic = u32::from_le_bytes(header_buf[0..4].try_into().unwrap());
        let _version = u32::from_le_bytes(header_buf[4..8].try_into().unwrap());
        let n_tensors = u32::from_le_bytes(header_buf[8..12].try_into().unwrap());
        let n_layers = u32::from_le_bytes(header_buf[12..16].try_into().unwrap());

        let is_valid_magic = magic == Self::QWN_MAGIC_QWN2 || magic == Self::QWN_MAGIC_COLI || magic == Self::QWN_MAGIC_QWN1;

        if !is_valid_magic {
            return ModelInfo {
                id: name.clone(),
                name,
                path: path.to_string_lossy().to_string(),
                path_alias,
                size_formatted,
                size_bytes,
                format: ".qwn container".into(),
                quantization: "Unknown".into(),
                compatibility_state: "invalid".into(),
                metadata_status: format!("Invalid: magic 0x{:08X} does not match QWN2/COLI/QWN1", magic),
                n_tensors: None,
                n_layers: None,
            };
        }

        // Parse first tensor entry descriptor to detect real quantization dtype
        // QwnTensorEntry: name[64], dtype(4), n_dims(4), shape[4](32), offset(8), size(8) = 120 bytes
        let mut entry_buf = [0u8; 120];
        let quantization = if n_tensors > 0 && file.read_exact(&mut entry_buf).is_ok() {
            let dtype = u32::from_le_bytes(entry_buf[64..68].try_into().unwrap());
            match dtype {
                4 => "TWLA 1.58-Bit Ternary".to_string(),
                3 => "HyperVSQ-2 (2.3125 bpw)".to_string(),
                5 => "TurboQuant (3.5 bpw KV)".to_string(),
                2 => "Q4_0 (4-bit linear)".to_string(),
                1 => "FP16 (Half Precision)".to_string(),
                0 => "FP32 (Single Precision)".to_string(),
                _ => format!("Custom QWN Dtype ({})", dtype),
            }
        } else {
            "Verified QWN Header (Zero Descriptor Table)".to_string()
        };

        ModelInfo {
            id: name.clone(),
            name,
            path: path.to_string_lossy().to_string(),
            path_alias,
            size_formatted,
            size_bytes,
            format: ".qwn container".into(),
            quantization,
            compatibility_state: "compatible".into(),
            metadata_status: "Verified 4KiB QWN Container Header".into(),
            n_tensors: Some(n_tensors),
            n_layers: Some(n_layers),
        }
    }

    pub fn inspect_gguf_file(path: &Path) -> ModelInfo {
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let size_bytes = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let size_gb = (size_bytes as f64) / (1024.0 * 1024.0 * 1024.0);

        ModelInfo {
            id: name.clone(),
            name,
            path: path.to_string_lossy().to_string(),
            path_alias: format!("external://{}", path.file_name().unwrap_or_default().to_string_lossy()),
            size_formatted: format!("{:.2} GB", size_gb),
            size_bytes,
            format: "GGUF (External Runtime)".into(),
            quantization: "External Quantization".into(),
            compatibility_state: "unavailable_by_default".into(),
            metadata_status: "External runtime disabled by default (--allow-external-runtime required)".into(),
            n_tensors: None,
            n_layers: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_valid_qwn_header_parsing() {
        let temp_dir = std::env::temp_dir().join("qwanto_test_valid_qwn");
        let _ = fs::create_dir_all(&temp_dir);
        let file_path = temp_dir.join("model_test.qwn");

        // Write 4096-byte valid QWN2 header
        let mut header = vec![0u8; 4096];
        header[0..4].copy_from_slice(&ModelRegistry::QWN_MAGIC_QWN2.to_le_bytes()); // "QWN2"
        header[4..8].copy_from_slice(&2u32.to_le_bytes()); // version = 2
        header[8..12].copy_from_slice(&1u32.to_le_bytes()); // n_tensors = 1
        header[12..16].copy_from_slice(&24u32.to_le_bytes()); // n_layers = 24

        // Write 1 tensor entry with dtype = 4 (TWLA_158)
        let mut entry = vec![0u8; 120];
        entry[0..4].copy_from_slice(b"root");
        entry[64..68].copy_from_slice(&4u32.to_le_bytes()); // dtype = 4

        let mut f = File::create(&file_path).unwrap();
        f.write_all(&header).unwrap();
        f.write_all(&entry).unwrap();
        f.flush().unwrap();

        let info = ModelRegistry::inspect_qwn_file(&file_path);
        assert_eq!(info.format, ".qwn container");
        assert_eq!(info.compatibility_state, "compatible");
        assert_eq!(info.quantization, "TWLA 1.58-Bit Ternary");
        assert_eq!(info.n_tensors, Some(1));
        assert_eq!(info.n_layers, Some(24));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_invalid_magic_fails_gracefully() {
        let temp_dir = std::env::temp_dir().join("qwanto_test_invalid_magic");
        let _ = fs::create_dir_all(&temp_dir);
        let file_path = temp_dir.join("corrupted.qwn");

        let mut header = vec![0u8; 4096];
        header[0..4].copy_from_slice(b"BAD!");

        let mut f = File::create(&file_path).unwrap();
        f.write_all(&header).unwrap();
        f.flush().unwrap();

        let info = ModelRegistry::inspect_qwn_file(&file_path);
        assert_eq!(info.compatibility_state, "invalid");
        assert!(info.metadata_status.contains("magic"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_non_ascii_and_spaces_path() {
        let temp_dir = std::env::temp_dir().join("qwanto test models 🚀");
        let _ = fs::create_dir_all(&temp_dir);
        let file_path = temp_dir.join("модель_test.qwn");

        let mut header = vec![0u8; 4096];
        header[0..4].copy_from_slice(&ModelRegistry::QWN_MAGIC_QWN2.to_le_bytes());
        header[8..12].copy_from_slice(&0u32.to_le_bytes()); // 0 tensors

        let mut f = File::create(&file_path).unwrap();
        f.write_all(&header).unwrap();

        let info = ModelRegistry::inspect_qwn_file(&file_path);
        assert_eq!(info.compatibility_state, "compatible");
        assert!(info.path.contains("qwanto test models 🚀"));

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_gguf_marked_unavailable_by_default() {
        let path = Path::new("test_model.gguf");
        let info = ModelRegistry::inspect_gguf_file(path);
        assert_eq!(info.compatibility_state, "unavailable_by_default");
        assert!(info.metadata_status.contains("--allow-external-runtime required"));
    }
}
