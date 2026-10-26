#!/usr/bin/env bash
# One-time setup on Linux / macOS:  bash setup.sh [--mirror] [--quick]
set -euo pipefail
cd "$(dirname "$0")"
IDX=(); QUICK=0
for a in "$@"; do
  case "$a" in
    --mirror) IDX=(-i https://pypi.tuna.tsinghua.edu.cn/simple) ;;
    --quick) QUICK=1 ;;
  esac
done
PY=""
for c in python3.12 python3.13 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Python 3.10-3.13 not found"; exit 1; }
[ -x .venv/bin/python ] || "$PY" -m venv .venv
V=.venv/bin/python
"$V" -m pip install --upgrade pip "${IDX[@]}"
"$V" -m pip install -r requirements.txt "${IDX[@]}"
"$V" -m pip install -e . --no-deps
"$V" -c "import torch; print('torch', torch.__version__, '| threads', torch.get_num_threads(), '| cuda', torch.cuda.is_available())"
"$V" -m pytest -q
if [ -z "$(find data -name '*_4d.nii*' 2>/dev/null | head -1)" ]; then
  echo; echo "No patient data yet. Put an ACDC patient folder into data/acdc/ (see data/README.md)."
elif [ "$QUICK" = 1 ]; then
  "$V" scripts/pipeline.py --preset quick
fi
echo; echo "Ready. Next:  source .venv/bin/activate && python scripts/pipeline.py"
