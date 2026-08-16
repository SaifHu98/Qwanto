# Model conversion and acquisition

The local gateway accepts explicit source formats for conversion: GGUF,
Safetensors, and PyTorch `.pt`, `.pth`, or supported PyTorch `.bin` files.
Unsupported formats such as ONNX, Keras/H5, and arbitrary binary files fail
with an explicit error. Conversion produces a validated `.qwn` container; the
gateway reports the actual output status and does not estimate size or speed.

Model discovery is validation-based. A model can be selected for native
inference only when its QWN structure passes validation, the local `qwnrun`
runtime is available, and the measured host resource plan says it fits. File
names alone never activate a model. See the [QWN format](qwn-format.md) and
[model acquisition design](model-acquisition-design.md).

Network acquisition is optional and requires an explicit user action. The UI
shows the source host, accepts an expected SHA-256 and size, and requires local
download/license confirmation. The gateway handles HTTPS host approval,
disk-space preflight, resumable `.part` transfers, checksum validation, and
atomic publication. The packaged desktop shell does not bundle this Python
gateway or its converter/downloader.
