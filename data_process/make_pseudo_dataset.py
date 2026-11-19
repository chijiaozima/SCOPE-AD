"""Generate fictional baseline examples independently of ADNI source files."""

import argparse
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from preparation.dataset import export_splits
from preparation.modalities import OUTPUT_COLUMNS
from preparation.mri import COMPOSITE_DEFINITIONS
from preparation.schema import LABELS


def block(title, values):
    if not values:
        return f"{title}: not available."
    bullets = [f"- {name}: {value}" for name, value in values.items()]
    return "\n".join([f"{title} source_visit=bl:", *bullets])


def generate():
    rng = random.Random(4090)
    rows = []
    for idx in range(60):
        # Values are drawn independently of the target. This is a parser and workflow fixture.
        label = LABELS[idx % len(LABELS)]
        demographics = {
            "birth_year": rng.randint(1940, 1965),
            "education_years": rng.randint(8, 20),
        }
        cognitive = {"MMSE": rng.randint(16, 30), "FAQ total": rng.randint(0, 10)}
        blood = {"Quanterix plasma GFAP (GFAP_Q)": rng.randint(75, 230)}
        csf = {"CSF amyloid beta 42": rng.randint(550, 1200)}
        mri = {
            name: round(
                rng.uniform(0.002, 0.009) if name.endswith("_icv") else rng.uniform(1.8, 3.3), 4
            )
            for name in COMPOSITE_DEFINITIONS
        }
        rows.append(
            {
                "row_id": str(idx + 1),
                "subject_id": f"SYN-{idx + 1:04d}",
                "visit": "bl",
                "cognitive_scales": block("Cognitive scales", cognitive),
                "demographic": block("Demographic", demographics),
                "medical_history": block("Medical history", {}),
                "apoe_genotype": block(
                    "APOE genotype",
                    {"APOE genotype": rng.choice(["3/3", "3/4", "2/3"])} if idx % 5 else {},
                ),
                "blood_biomarkers": block("Blood biomarkers", blood if idx % 4 else {}),
                "csf_biomarkers": block("CSF biomarkers", csf if idx % 4 != 1 else {}),
                "structural_mri": "Structural MRI AD composites:\n"
                + "\n".join(f"- {k}: {v}" for k, v in mri.items()),
                "amyloid_pet": block(
                    "Amyloid PET",
                    {"amyloid summary SUVR": round(rng.uniform(0.7, 1.6), 3)} if idx % 3 else {},
                ),
                "tau_pet": block(
                    "Tau PET",
                    (
                        {"tau meta-temporal SUVR": round(rng.uniform(0.7, 1.7), 3)}
                        if idx % 5 < 2
                        else {}
                    ),
                ),
                "diagnose": f"Diagnose source_visit=current:\n- raw_DIAGNOSIS: {1 + LABELS.index(label)}\n- mapped_label: {label}",
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New directory for the synthetic dataset"
    )
    output = parser.parse_args().output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error("Output already exists. Select a new directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as work:
        stage = Path(work) / "dataset"
        stage.mkdir()
        metadata = export_splits(generate(), stage, synthetic=True)
        stage.rename(output)
    print(f"Synthetic dataset: {output}. Counts: {metadata['counts']}")


if __name__ == "__main__":
    main()
