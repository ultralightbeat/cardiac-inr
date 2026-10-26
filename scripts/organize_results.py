"""One-time: move the runs of 26 Oct 2026 (patient051) from the old flat layout into the final one.

    results/patient051/experiments/<run>/   every experiment, with a clear name
    results/patient051/final/heart4d/       the 3D/4D heart (re-render after this script)
    results/patient051/SUMMARY.md           the tables of the report in one file
    models/patient051/field.pt              the final field (+ field.json)

Safe to run twice: folders that are already moved are skipped. Nothing is deleted except the
obsolete preview render (heart4d_preview) and the quick smoke test.

    python scripts/organize_results.py            # show what would happen
    python scripts/organize_results.py --apply    # do it
"""

from __future__ import annotations

import argparse
import os
import shutil

P = "patient051"
MOVES = [  # old folder under results/ -> new folder under results/patient051/experiments/
    ("patient051", "main", "E1 + E2 (every 2nd frame hidden), raw / scaled 0.25 / periodic, 4000 steps"),
    ("e2_long", "stride2_s0.25_20k", "E2 every 2nd frame hidden, scaled 0.25, 20000 steps"),
    ("e2_stride4", "stride4_s0.25", "every 4th frame kept, scaled 0.25"),
    ("e2_stride4_w512", "stride4_s0.25_w512", "same, 512 neurons"),
    ("e2_stride4_s0.1", "stride4", "every 4th frame kept, scaled 0.1 (the reference run)"),
    ("e2_stride4_s0.05", "stride4_s0.05", "every 4th frame kept, scaled 0.05"),
    ("e2_stride4_relu", "stride4_relu", "ReLU control, same settings as stride4"),
    ("e2_stride4_full", "stride4_fullimage", "no crop: whole scan, 86x compression"),
]
DROP = ["patient051_quick", "heart4d_preview"]

SUMMARY_PARTS = [
    ("E1 hidden slices and E2 every second frame hidden (4000 steps)", "main/results.md"),
    ("E2 every second frame hidden, 20000 steps", "stride2_s0.25_20k/results.md"),
    ("Blend on every second frame hidden", "stride2_s0.25_20k/blend.md"),
    ("Every 4th frame kept: time scale 0.25", "stride4_s0.25/results.md"),
    ("Every 4th frame kept: time scale 0.25, 512 neurons", "stride4_s0.25_w512/results.md"),
    ("Every 4th frame kept: time scale 0.1 (reference)", "stride4/results.md"),
    ("Every 4th frame kept: time scale 0.05", "stride4_s0.05/results.md"),
    ("ReLU control (same settings as the reference)", "stride4_relu/results.md"),
    ("Blend 0.5 SIREN + 0.5 interpolation (reference)", "stride4/blend.md"),
    ("Whole scan without crop (86x compression)", "stride4_fullimage/results.md"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    act = "MOVE" if a.apply else "would move"
    exp = os.path.join("results", P, "experiments")

    # the old run folder results/patient051 has the same name as the new patient folder:
    # rename it first (old layout = results.json directly inside it)
    legacy = os.path.join("results", P)
    if os.path.exists(os.path.join(legacy, "results.json")):
        print(f"{act}: {legacy} -> results/_{P}_main (temporary)")
        if a.apply:
            shutil.move(legacy, os.path.join("results", f"_{P}_main"))
    moves = [(f"_{P}_main" if old == P else old, new, why) for old, new, why in MOVES]
    if not a.apply:
        moves = MOVES
    for old, new, _ in moves:
        src, dst = os.path.join("results", old), os.path.join(exp, new)
        if os.path.isdir(src) and not os.path.exists(dst):
            print(f"{act}: {src} -> {dst}")
            if a.apply:
                os.makedirs(exp, exist_ok=True)
                shutil.move(src, dst)
    old_final = os.path.join("results", "heart4d")
    new_final = os.path.join("results", P, "final", "heart4d")
    mdir = os.path.join("models", P)
    if os.path.isdir(old_final):
        for f in ("field.pt", "field.json"):
            if os.path.exists(os.path.join(old_final, f)) and not os.path.exists(os.path.join(mdir, f)):
                print(f"{act}: {os.path.join(old_final, f)} -> {mdir}")
                if a.apply:
                    os.makedirs(mdir, exist_ok=True)
                    shutil.move(os.path.join(old_final, f), os.path.join(mdir, f))
        if not os.path.exists(new_final):
            print(f"{act}: {old_final} -> {new_final}")
            if a.apply:
                os.makedirs(os.path.dirname(new_final), exist_ok=True)
                shutil.move(old_final, new_final)
    for old in DROP:
        src = os.path.join("results", old)
        if os.path.isdir(src):
            print(("DELETE" if a.apply else "would delete") + f": {src} (obsolete)")
            if a.apply:
                shutil.rmtree(src)

    if a.apply:
        lines = [f"# {P}: all experiments", "",
                 "PSNR in dB on data hidden from the network; 'heart' = inside the dilated expert labels. "
                 "Run folders: results/patient051/experiments/. Final 3D/4D heart: final/heart4d/.", ""]
        notes = {new: why for _, new, why in MOVES}
        for title, rel in SUMMARY_PARTS:
            path = os.path.join(exp, rel)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    body = [ln for ln in f.read().splitlines() if not ln.startswith("# ")]
                run = rel.split("/")[0]
                lines += [f"## {title}", "", f"`experiments/{run}`: {notes.get(run, '')}", ""] + body + [""]
        fs = os.path.join(new_final, "summary.md")
        if os.path.exists(fs):
            with open(fs, encoding="utf-8") as f:
                body = [ln for ln in f.read().splitlines() if not ln.startswith("# ")]
            lines += ["## Final field on all data: volumes and ejection fraction", ""] + body + [""]
        with open(os.path.join("results", P, "SUMMARY.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote results/{P}/SUMMARY.md")
        print("\nnext: python scripts/render_4d.py --patient data/acdc/patient051   (re-render with the latest 3D fixes)")
    else:
        print("\nnothing changed yet; run again with --apply")


if __name__ == "__main__":
    main()
