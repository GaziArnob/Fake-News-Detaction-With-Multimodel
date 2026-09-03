"""Run the notebook's CISF cell against cached project features for validation."""

from __future__ import annotations

import json
import os
from pathlib import Path


notebook_path = Path(__file__).resolve().parents[2] / "notebook.ipynb"
os.chdir(notebook_path.parent)
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
namespace: dict[str, object] = {
    "__name__": "__main__",
    "__file__": str(notebook_path),
}
targets = (
    "BASE_PATH = Path",
    "records = []",
    "OCR_LANGUAGES =",
    "CISF_NEIGHBORS =",
)

for index, cell in enumerate(notebook["cells"]):
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "code" and any(target in source for target in targets):
        exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
