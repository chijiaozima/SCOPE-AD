"""Apply the historical MRI and biomarker availability criteria."""

from __future__ import annotations

import math

import pandas as pd


MRI_COMPOSITES = (
    "hippocampal_volume_icv",
    "entorhinal_thickness",
    "entorhinal_volume_icv",
    "medial_temporal_lobe_volume_icv",
    "medial_temporal_lobe_thickness",
    "ad_signature_cortical_thickness",
    "temporal_neocortex_thickness",
    "posterior_temporoparietal_thickness",
)


MISSING_TEXT_VALUES = {"", "na", "n/a", "nan", "none", "null"}


def parse_structural_mri_composites(text: object) -> dict[str, str]:
    """Parse compact structural_mri composite text into name/value strings."""
    parsed: dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        name, separator, value = line[2:].partition(":")
        if separator:
            parsed[name.strip()] = value.strip()
    return parsed


def is_missing_value(value: object) -> bool:
    """Return True for empty, NaN-like, or non-finite values."""
    text = str(value or "").strip()
    if text.lower() in MISSING_TEXT_VALUES:
        return True
    try:
        return not math.isfinite(float(text))
    except ValueError:
        return False


def contains_not_available(value: object) -> bool:
    """Return True when a modality text field indicates not available."""
    text = str(value or "").strip().lower()
    return text in MISSING_TEXT_VALUES or "not available" in text


def count_missing_structural_composites(mri_text: object) -> int:
    """Count missing values among the eight AD MRI composites."""
    parsed = parse_structural_mri_composites(mri_text)
    return sum(is_missing_value(parsed.get(composite, "NaN")) for composite in MRI_COMPOSITES)


def build_keep_mask(
    dataframe: pd.DataFrame,
    mri_column: str,
    blood_column: str,
    csf_column: str,
    structural_missing_threshold: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Build keep mask and the two delete-condition masks."""
    for column in (mri_column, blood_column, csf_column):
        if column not in dataframe.columns:
            raise ValueError(f"Required column is missing: {column}")

    structural_missing_counts = dataframe[mri_column].apply(count_missing_structural_composites)
    delete_structural = structural_missing_counts >= structural_missing_threshold
    delete_blood_csf = dataframe[blood_column].apply(contains_not_available) & dataframe[
        csf_column
    ].apply(contains_not_available)
    keep = ~(delete_structural | delete_blood_csf)
    return keep, delete_structural, delete_blood_csf
