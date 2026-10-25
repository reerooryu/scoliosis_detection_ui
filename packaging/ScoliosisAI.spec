# PyInstaller recipe for the all-in-one desktop app (app + AI model code).
# The model file is NOT included; testers get it separately.
#
# Build (from the repository root, with the app, server and build
# requirements installed -- see .github/workflows/release.yml):
#   pyinstaller packaging/ScoliosisAI.spec --noconfirm
# Output: dist/ScoliosisAI/ (Windows, Linux) or dist/ScoliosisAI.app (Mac).

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata, get_package_paths

ROOT = Path(SPECPATH).parent
NAME = "ScoliosisAI"

datas, binaries, hiddenimports = [], [], []

# DICOM decoders: pylibjpeg finds its decoder plugins through package
# metadata, so the metadata must be bundled along with the code.
# (import name, distribution name)
for package, distribution in (
    ("pylibjpeg", "pylibjpeg"),
    ("libjpeg", "pylibjpeg-libjpeg"),
    ("openjpeg", "pylibjpeg-openjpeg"),
):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas + copy_metadata(distribution)
    binaries += package_binaries
    hiddenimports += package_hidden

# torchvision's compiled operators (NMS, RoIAlign) are loaded by file name at
# run time, so PyInstaller does not see them. Recent versions name them
# "_C_stable" and "image_stable", which PyInstaller's own torchvision hook
# misses; without them the model fails with "torchvision::nms does not exist".
_, torchvision_dir = get_package_paths("torchvision")
for pattern in ("*.so", "*.pyd", "*.dll", "*.dylib"):
    binaries += [(str(path), "torchvision") for path in Path(torchvision_dir).glob(pattern)]

# Detectron2 registers its model parts when its modules are imported; take
# them all so nothing is missing at run time.
hiddenimports += collect_submodules("detectron2")
hiddenimports += ["modules.inference"]   # imported only when Local mode first runs

a = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "IPython", "jupyter", "notebook", "pytest"],
    # torch.jit.script reads the Python source of the functions it compiles,
    # so these packages are bundled as .py files as well as compiled.
    module_collection_mode={
        "detectron2": "pyz+py",
        "fvcore": "pyz+py",
        "timm": "pyz+py",
        "torchvision": "pyz+py",
    },
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    console=False,            # a normal window, no terminal
    upx=False,
    target_arch=None,
)

coll = COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{NAME}.app",
        bundle_identifier="org.scoliosisai.app",
        info_plist={
            "CFBundleDisplayName": "Scoliosis AI",
            "NSHighResolutionCapable": True,
        },
    )
