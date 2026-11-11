"""Run a step script block-by-block, saving each block's stdout.

A step file is split on lines starting with '# %% <name>'. Each block is executed in a
shared namespace and its printed output is stored in reports/outputs/<step>/<name>.txt.
Steps are also valid standalone Python scripts (the markers are just comments).
"""
import contextlib
import io
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

path = Path(sys.argv[1])
step = path.stem
outdir = ROOT / "reports" / "outputs" / step
outdir.mkdir(parents=True, exist_ok=True)
for old in outdir.glob("*.txt"):
    old.unlink()

blocks, name, buf = [], "header", []
for line in path.read_text().splitlines():
    m = re.match(r"# %%\s+(\S+)", line)
    if m:
        blocks.append((name, "\n".join(buf)))
        name, buf = m.group(1), []
    else:
        buf.append(line)
blocks.append((name, "\n".join(buf)))

ns = {"__name__": "__main__", "__file__": str(path)}
t_all = time.time()
for name, src in blocks:
    if not src.strip():
        continue
    out = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(src, f"{step}:{name}", "exec"), ns)
    except Exception:
        (outdir / f"{name}.txt").write_text(out.getvalue() + traceback.format_exc())
        print(f"!! {step}:{name} FAILED\n{traceback.format_exc()}")
        sys.exit(1)
    (outdir / f"{name}.txt").write_text(out.getvalue().rstrip() + "\n")
    print(f"  ok  {step}:{name}  ({time.time() - t0:.1f}s)")
print(f"{step} finished in {time.time() - t_all:.0f}s")
