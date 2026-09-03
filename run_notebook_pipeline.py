"""Execute the notebook's Python cells with the project virtual environment.

The first cell is intentionally skipped because packages were verified as installed.
This runner executes the remaining cells in order, so `extracted_features.csv` is
checkpointed exactly as described in notebook.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path(__file__).with_name("notebook.ipynb")


def main() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__main__", "__file__": str(NOTEBOOK_PATH)}

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        if source.lstrip().startswith("!"):
            print("Step 1: packages already verified; skipping notebook pip command.", flush=True)
            continue

        print(f"Executing notebook code cell {index}...", flush=True)
        exec(compile(source, f"{NOTEBOOK_PATH.name}:cell_{index}", "exec"), namespace)


if __name__ == "__main__":
    main()
