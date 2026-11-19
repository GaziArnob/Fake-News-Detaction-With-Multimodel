"""One-off: wrap the adaptation-figure block of make_journal_figures.py in a function
so it can be drawn at two font scales (IEEE two-column and Springer single-column)."""
import io
import re
from pathlib import Path

p = Path(__file__).with_name("make_journal_figures.py")
L = io.open(p, encoding="utf-8").read().split("\n")
a = next(i for i, l in enumerate(L) if l.startswith("fig, axes = plt.subplots(1, 3"))
b = next(i for i, l in enumerate(L) if l.startswith('print("\\nwrote fig_j_adaptation.png")'))
block = L[a:b + 1]
txt = "\n".join(block)
# scale every hard-coded font size
txt = re.sub(r"fontsize=([0-9.]+)", lambda m: f"fontsize={m.group(1)} * S", txt)
txt = txt.replace("figsize=(7.1, 2.55)", "figsize=size")
txt = txt.replace('OUT / "fig_j_adaptation.png"', 'OUT / f"fig_j_adaptation{tag}.png"')
txt = txt.replace('print("\\nwrote fig_j_adaptation.png")', 'print(f"\\nwrote fig_j_adaptation{tag}.png")')
txt = txt.replace('ax.set_xlabel("accuracy gain from adaptation (points)")',
                  'ax.set_xlabel("accuracy gain (points)", fontsize=8 * S)')
txt = txt.replace('ax.set_xlabel("AUROC gain from adaptation")', 'ax.set_xlabel("AUROC gain", fontsize=8 * S)')
txt = txt.replace('ax.set_ylabel("AUROC")', 'ax.set_ylabel("AUROC", fontsize=8 * S)')
txt = re.sub(r'(ax\.set_title\([^\n]*?loc="left")', r'\1, fontsize=8.5 * S', txt)
txt = txt.replace("ax.tick_params", "ax.tick_params")
new = ["def draw_adaptation(S, size, tag):",
       "    plt.rcParams.update({'xtick.labelsize': 7 * S, 'ytick.labelsize': 7 * S})"]
new += ["    " + l if l.strip() else l for l in txt.split("\n")]
new += ["", "", 'draw_adaptation(1.0, (7.1, 2.55), "")',
        'draw_adaptation(1.45, (7.1, 3.7), "_springer")']
L[a:b + 1] = new
io.open(p, "w", encoding="utf-8").write("\n".join(L))
print("wrapped", b - a + 1, "lines")
