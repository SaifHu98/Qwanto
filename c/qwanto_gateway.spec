from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules


SPEC_DIR = Path(SPECPATH).resolve()
hiddenimports = [
    "backends",
    "capabilities",
    "doctor",
    "model_acquisition",
    "orchestrator",
    "resource_plan",
    "tools.qwn_convert",
]
hiddenimports.extend(collect_submodules("tools"))

a = Analysis(
    [str(SPEC_DIR / "openai_server.py")],
    pathex=[str(SPEC_DIR), str(SPEC_DIR / "tools")],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "transformers", "safetensors", "pandas"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="qwanto-gateway",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
