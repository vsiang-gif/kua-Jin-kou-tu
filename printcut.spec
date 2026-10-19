# PyInstaller spec: pyinstaller printcut.spec --noconfirm
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas, binaries, hiddenimports = [], [], []
for pkg in ("vtracer", "resvg_py"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
binaries += collect_dynamic_libs("onnxruntime")
hiddenimports += ["onnxruntime", "scipy.ndimage"]

a = Analysis(
    ["run_printcut.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pandas", "torch", "PyQt5", "PyQt6"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PrintCut",
    console=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PrintCut", upx=False)
