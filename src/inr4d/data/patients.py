"""Finding patients and the standard output folders.

Every script takes --patient, which can be
  * a patient folder in ACDC layout (data/acdc/patient051), or
  * any 4D NIfTI file [H, W, D, T] (labels and Info.cfg are then optional), or
  * omitted: the first patient found under data/ is used.

Outputs:  results/<patient>/experiments/<run>/   tables, figures, json of each experiment
          results/<patient>/final/heart4d/      the 3D/4D heart from the final field
          results/<patient>/SUMMARY.md          all tables in one place
          models/<patient>/field.pt             the final trained neural field (+ field.json)
"""

from __future__ import annotations

import glob
import os


def patient_name(path: str) -> str:
    path = os.path.normpath(path)
    if os.path.isfile(path):
        base = os.path.basename(path)
        for ext in (".nii.gz", ".nii"):
            if base.endswith(ext):
                base = base[: -len(ext)]
        return base[:-3] if base.endswith("_4d") else base
    return os.path.basename(path)


def patient_dir(path: str) -> str:
    """Folder that may hold labels (*_gt.nii) and Info.cfg."""
    return os.path.dirname(path) if os.path.isfile(path) else path


def find_patients(root: str = "data") -> list[str]:
    """Folders under root that contain a 4D NIfTI (*_4d.nii / *_4d.nii.gz), sorted."""
    hits = glob.glob(os.path.join(root, "**", "*_4d.nii"), recursive=True)
    hits += glob.glob(os.path.join(root, "**", "*_4d.nii.gz"), recursive=True)
    return sorted({os.path.dirname(h) for h in hits})


def resolve_patient(arg: str | None, root: str = "data") -> str:
    if arg:
        if not os.path.exists(arg):
            raise SystemExit(f"--patient {arg}: not found")
        return arg
    found = find_patients(root)
    if not found:
        raise SystemExit(f"No patient given and no *_4d.nii(.gz) found under {root}/. "
                         "Put a patient folder into data/ (see data/README.md) or pass --patient.")
    print(f"using patient {found[0]} (first of {len(found)} found under {root}/; pass --patient to choose)")
    return found[0]


def results_dir(patient: str, run: str) -> str:
    return os.path.normpath(os.path.join("results", patient_name(patient), run))


def model_dir(patient: str) -> str:
    return os.path.join("models", patient_name(patient))
