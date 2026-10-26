import os

from inr4d.data import patients as P


def test_patient_name_from_folder_and_file(tmp_path):
    d = tmp_path / "patient007"
    d.mkdir()
    f = d / "patient007_4d.nii.gz"
    f.write_bytes(b"")
    assert P.patient_name(str(d)) == "patient007"
    assert P.patient_name(str(f)) == "patient007"
    assert P.patient_dir(str(f)) == str(d)


def test_find_patients(tmp_path):
    for name in ("patient002", "patient001"):
        d = tmp_path / "acdc" / name
        d.mkdir(parents=True)
        (d / f"{name}_4d.nii").write_bytes(b"")
    found = P.find_patients(str(tmp_path))
    assert [os.path.basename(x) for x in found] == ["patient001", "patient002"]
    assert P.results_dir(found[0], "final/heart4d").endswith(os.path.join("patient001", "final", "heart4d"))
