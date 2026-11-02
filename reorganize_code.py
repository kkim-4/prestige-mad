"""Second tidy-up: code into folders. Shows the plan; nothing changes until you pass --go.

    python3 reorganize_code.py          # dry run
    python3 reorganize_code.py --go     # do it

Layout afterwards:
  mad_dual.py, mad_gemini.py   top level: the main script, and the shared helpers every analysis imports
  analysis/                    every analysis script (signal_check, vote_check, mixed_report, ...)
  scripts/                     every run_*.sh, serve_models.sh, make_gpqa.py, make_math_extra.py
  archived/                    every old mad_gemini_v*.py (v5 included), run_experiment.sh, mad_dual_v2.py

Edits so everything still works when run from the repo root:
  - each moved .sh starts with `cd` to the repo root, so `bash scripts/run_qwen.sh` works from anywhere;
    calls inside point to scripts/... and analysis/...; mad_gemini_v5.py calls become mad_dual.py
    (mad_dual.py does everything v5 did; v5 itself is kept in archived/ for the record)
  - each moved .py finds mad_gemini.py in the repo root; usage lines in docstrings get the new path
Tracked files move with `git mv` (git sees renames); untracked files move with a plain rename.
"""
import os, re, subprocess, sys

GO = "--go" in sys.argv
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if not (os.path.exists("mad_gemini.py") and os.path.exists("mad_dual.py")):
    sys.exit("run this inside the mad_gemini folder")

KEEP = {"mad_dual.py", "mad_gemini.py", "reorganize_code.py"}
top_py = sorted(f for f in os.listdir(".") if f.endswith(".py") and os.path.isfile(f))
top_sh = sorted(f for f in os.listdir(".") if f.endswith(".sh") and os.path.isfile(f))

archive = [f for f in top_py if re.fullmatch(r"mad_gemini_v\d+\.py|mad_gemini_independent_personas\.py|mad_dual_v\d+\.py", f)]
archive += [f for f in ("run_experiment.sh", "reorganize.sh") if f in top_sh]
scripts = [f for f in top_sh if f not in archive] + [f for f in top_py if f.startswith("make_")]
analysis = [f for f in top_py if f not in KEEP and f not in archive and f not in scripts]
A = {f[:-3] for f in analysis} | {f[:-3] for f in os.listdir("analysis") if f.endswith(".py")} \
    if os.path.isdir("analysis") else {f[:-3] for f in analysis}
S = {f for f in scripts} | (set(os.listdir("scripts")) if os.path.isdir("scripts") else set())


def tracked(f):
    return subprocess.run(["git", "ls-files", "--error-unmatch", f], capture_output=True).returncode == 0


def move(f, d):
    dst = os.path.join(d, f)
    if os.path.exists(dst):
        print(f"  SKIP {f} (already in {d}/)"); return None
    print(f"  {f} -> {d}/")
    if GO:
        os.makedirs(d, exist_ok=True)
        if tracked(f):
            subprocess.run(["git", "mv", f, dst], check=True)
        else:
            os.rename(f, dst)
    return dst


def fix_py(text):
    old = "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))"
    new = "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root"
    if old in text:
        text = text.replace(old, new)
    elif "mad_gemini import" in text and "repo root" not in text:
        text = re.sub(r"^(from mad_gemini import|import mad_gemini)", "import os, sys\n" + new + r"\n\1",
                      text, count=1, flags=re.M)
    return re.sub(r"python3 (?!analysis/)(\w+)\.py", lambda m: f"python3 analysis/{m[1]}.py" if m[1] in A
                  else m[0], text)


def fix_sh(text):
    text = text.replace("mad_gemini_v5.py", "mad_dual.py")
    text = re.sub(r"python3 (?!analysis/|scripts/)(\w+)\.py",
                  lambda m: f"python3 analysis/{m[1]}.py" if m[1] in A
                  else (f"python3 scripts/{m[1]}.py" if f"{m[1]}.py" in S else m[0]), text)
    text = re.sub(r"bash (?!scripts/)(\w+\.sh)", lambda m: f"bash scripts/{m[1]}" if m[1] in S else m[0], text)
    text = text.replace("vllm_logs", "logs/vllm")
    if 'cd "$(dirname "$0")/.."' not in text:
        text = re.sub(r"^(set -[euo]+ pipefail)$", r'\1\ncd "$(dirname "$0")/.."   # run from the repo root',
                      text, count=1, flags=re.M)
    return text


print("== mad_dual.py")
if os.path.exists("mad_dual_v2.py") and "seat-shuffle" in open("mad_dual_v2.py").read() \
        and "seat-shuffle" not in open("mad_dual.py").read():
    print("  mad_dual_v2.py replaces mad_dual.py (it has --seat-shuffle and dual_conf)")
    if GO:
        os.replace("mad_dual_v2.py", "mad_dual.py")
    archive = [f for f in archive if f != "mad_dual_v2.py"]
elif "seat-shuffle" not in open("mad_dual.py").read():
    print("  WARNING: mad_dual.py lacks --seat-shuffle; save the latest version here first")

print("== old versions -> archived/")
for f in archive:
    move(f, "archived")
print("== run scripts and data makers -> scripts/")
moved_sh = [move(f, "scripts") for f in scripts]
print("== analysis scripts -> analysis/")
moved_py = [move(f, "analysis") for f in analysis]

print("== path fixes inside moved files")
for dst, fixer in [(d, fix_sh if d.endswith(".sh") else fix_py) for d in moved_sh + moved_py if d]:
    src = dst if GO else os.path.basename(dst)
    t = open(src).read()
    n = fixer(t)
    if n != t:
        print(f"  edit {dst}")
        if GO:
            open(dst, "w").write(n)

if GO:
    for d in ("scripts",):
        for f in os.listdir(d):
            if f.endswith(".sh"):
                os.chmod(os.path.join(d, f), 0o755)
    print("\nDone. Check with:  git status   and a quick test:  python3 analysis/signal_check.py --help")
else:
    print("\nDry run only. If the plan looks right:  python3 reorganize_code.py --go")
