"""Run the notebook's baseline and CISF evaluation cells using cached features."""

from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_DIR / "notebook.ipynb"
os.chdir(PROJECT_DIR)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
namespace: dict[str, object] = {"__name__": "__main__", "__file__": str(NOTEBOOK_PATH)}
targets = (
    "from pathlib import Path",
    "OCR_LANGUAGES =",
    "MODEL_FEATURES =",
    "CISF_NEIGHBORS =",
)

for index, cell in enumerate(notebook["cells"]):
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "code" and any(target in source for target in targets):
        print(f"Running notebook cell {index}", flush=True)
        exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
