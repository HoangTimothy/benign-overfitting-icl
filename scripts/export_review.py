"""Export every committed change from the pristine Garg baseline for review."""

import difflib
import html
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "garg-2022-baseline"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


out = ROOT / "review" / "generated"
out.mkdir(parents=True, exist_ok=True)
(out / "all-changes.patch").write_bytes(git("diff", "--binary", BASE, "HEAD"))
(out / "summary.txt").write_bytes(git("diff", "--stat", BASE, "HEAD"))
tables = []
for name in git("diff", "--name-only", BASE, "HEAD").decode().splitlines():
    def content(ref):
        result = subprocess.run(["git", "show", f"{ref}:{name}"], cwd=ROOT, capture_output=True)
        return result.stdout.decode("utf-8", errors="replace").splitlines()
    old, new = content(BASE), content("HEAD")
    table = difflib.HtmlDiff(wrapcolumn=110).make_table(
        old, new, fromdesc=html.escape(f"Garg: {name}"),
        todesc=html.escape(f"HEAD: {name}"), context=True, numlines=4,
    )
    tables.append(f"<h2>{html.escape(name)}</h2>" + table)
head = git("rev-parse", "HEAD").decode().strip()
(out / "line-by-line.html").write_text(
    '<!doctype html><meta charset="utf-8"><title>Garg modernization review</title>'
    '<style>body{font-family:system-ui}table{font-family:monospace;font-size:12px;border-collapse:collapse}'
    'td{padding:2px 5px}.diff_add{background:#cfc}.diff_sub{background:#fcc}'
    '.diff_chg{background:#ffc}.diff_header{background:#eee}</style>'
    f'<h1>Line-by-line review</h1><p>{BASE} → {head}</p>' + "\n".join(tables),
    encoding="utf-8",
)
print(out)
