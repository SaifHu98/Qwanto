# 📦 Qwanto Native Packaging & Platform Support Matrix

## 1. Supported Platform Matrix

| Platform / OS | Hardware Architecture | Inference Backend | Status | Validation Evidence |
|---|---|---|:---:|---|
| **Windows 11 (24H2)** | x86_64 (AMD Ryzen 9 / Intel Core) | **NVIDIA CUDA (SM89) + AVX-VNNI / AVX-512 + NVMe mmap** | 🟢 **VERIFIED** | Live host benchmark (16 cores/32T, RTX 5070 Ti 12GB) |
| **Linux (Ubuntu 22.04+)** | x86_64 | **AVX-512 / AVX2 + OpenMP SIMD** | 🟢 **VERIFIED** | GitHub Actions CI (`make test-c` & Pytest) |
| **Linux (CUDA)** | x86_64 (NVIDIA GPUs) | **CUDA 12.x + BitDecoding Tensor Cores** | 🟡 *In-CI Syntax Checked* | CI `nvcc -arch=sm_80` syntax validated |
| **macOS (Sonoma+)** | Apple Silicon (M1/M2/M3/M4) | **Metal Compute Shaders (`backend_metal.mm`)** | ⚪ *Experimental / Unverified* | Marked experimental pending local macOS runner |

---

## 2. Platform Packaging Instructions

### 🪟 Windows (MSVC / Clang)

#### 1. Compile Native Engine (`qwnrun.exe`)
```powershell
cd c
# Build optimized native binary with AVX-VNNI / AVX-512 support:
cl /O2 /W4 /D_CRT_SECURE_NO_WARNINGS /Fe:qwnrun.exe qwnrun.c qwanto_decode.c qwanto_gpu.c qwanto_autopilot.c /link /out:qwnrun.exe
```

#### 2. Package Tauri Desktop Application
```powershell
cd desktop
npm run build
npx tauri build --target x86_64-pc-windows-msvc
```
*Output: `desktop/src-tauri/target/release/bundle/msi/Qwanto_0.1.0_x64_en-US.msi`*

---

### 🐧 Linux (Debian / Ubuntu / Fedora)

#### 1. Compile Native Engine
```bash
cd c
make qwnrun
make test-c
```

#### 2. Package Tauri Desktop Application
```bash
cd desktop
npm run build
cargo tauri build --bundles deb,appimage
```
*Output: `desktop/src-tauri/target/release/bundle/appimage/qwanto_0.1.0_amd64.AppImage`*

---

### 🍎 macOS (Apple Silicon)

#### 1. Compile Native Engine with Metal Backend
```bash
cd c
clang -O3 -framework Metal -framework Foundation qwnrun.c backend_metal.mm -o qwnrun
```

#### 2. Package Tauri Desktop Application
```bash
cd desktop
npm run build
cargo tauri build --bundles dmg
```
*Output: `desktop/src-tauri/target/release/bundle/dmg/Qwanto_0.1.0_aarch64.dmg`*
