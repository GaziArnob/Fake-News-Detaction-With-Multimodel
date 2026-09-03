"""Execute notebook Steps 9 and 10 using the stored extracted features."""

from __future__ import annotations

import json
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_DIR / "notebook.ipynb"
os.chdir(PROJECT_DIR)
os.environ.setdefault("MPLBACKEND", "Agg")

notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
namespace: dict[str, object] = {"__name__": "__main__", "__file__": str(NOTEBOOK_PATH)}
targets = (
    "BASE_PATH = Path",
    "TEXT_TEST_SIZE =",
    "MODEL_METRICS_CSV =",
    "CROSS_VALIDATION_FOLDS =",
)

for index, cell in enumerate(notebook["cells"]):
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "code" and any(target in source for target in targets):
        print(f"Running notebook cell {index}", flush=True)
        exec(compile(source, f"notebook-cell-{index}", "exec"), namespace)
