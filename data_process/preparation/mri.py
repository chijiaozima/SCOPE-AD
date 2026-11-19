"""Compute the eight structural MRI summaries used by the 4090pc cohort."""

from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np


NUMERIC_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


BULLET_RE = re.compile(r"^\s*[-*•]+\s*")


ICV_VARIABLE = "ICV volume"


COMPOSITE_DEFINITIONS = {
    # Bilateral hippocampal volume normalized by intracranial volume.
    "hippocampal_volume_icv": {
        "operation": "sum_icv",
        "variables": [
            "Left Hippocampus subcortical volume",
            "Right Hippocampus subcortical volume",
        ],
    },
    # Mean bilateral entorhinal cortical thickness.
    "entorhinal_thickness": {
        "operation": "mean",
        "variables": [
            "Left Entorhinal average cortical thickness",
            "Right Entorhinal average cortical thickness",
        ],
    },
    # Bilateral entorhinal cortical volume normalized by intracranial volume.
    "entorhinal_volume_icv": {
        "operation": "sum_icv",
        "variables": [
            "Left Entorhinal cortical volume",
            "Right Entorhinal cortical volume",
        ],
    },
    # Hippocampal, entorhinal, and parahippocampal volume normalized by ICV.
    "medial_temporal_lobe_volume_icv": {
        "operation": "sum_icv",
        "variables": [
            "Left Hippocampus subcortical volume",
            "Right Hippocampus subcortical volume",
            "Left Entorhinal cortical volume",
            "Right Entorhinal cortical volume",
            "Left Parahippocampal cortical volume",
            "Right Parahippocampal cortical volume",
        ],
    },
    # Mean entorhinal and parahippocampal cortical thickness.
    "medial_temporal_lobe_thickness": {
        "operation": "mean",
        "variables": [
            "Left Entorhinal average cortical thickness",
            "Right Entorhinal average cortical thickness",
            "Left Parahippocampal average cortical thickness",
            "Right Parahippocampal average cortical thickness",
        ],
    },
    # Mean cortical thickness across AD-signature cortical regions.
    "ad_signature_cortical_thickness": {
        "operation": "mean",
        "variables": [
            "Left Entorhinal average cortical thickness",
            "Right Entorhinal average cortical thickness",
            "Left Inferior Temporal average cortical thickness",
            "Right Inferior Temporal average cortical thickness",
            "Left Middle Temporal average cortical thickness",
            "Right Middle Temporal average cortical thickness",
            "Left Temporal Pole average cortical thickness",
            "Right Temporal Pole average cortical thickness",
            "Left Fusiform average cortical thickness",
            "Right Fusiform average cortical thickness",
            "Left Precuneus average cortical thickness",
            "Right Precuneus average cortical thickness",
            "Left Inferior Parietal average cortical thickness",
            "Right Inferior Parietal average cortical thickness",
            "Left Superior Parietal average cortical thickness",
            "Right Superior Parietal average cortical thickness",
            "Left Supramarginal average cortical thickness",
            "Right Supramarginal average cortical thickness",
            "Left Superior Frontal average cortical thickness",
            "Right Superior Frontal average cortical thickness",
            "Left Rostral Middle Frontal average cortical thickness",
            "Right Rostral Middle Frontal average cortical thickness",
            "Left Caudal Middle Frontal average cortical thickness",
            "Right Caudal Middle Frontal average cortical thickness",
        ],
    },
    # Mean thickness across lateral temporal neocortical regions.
    "temporal_neocortex_thickness": {
        "operation": "mean",
        "variables": [
            "Left Inferior Temporal average cortical thickness",
            "Right Inferior Temporal average cortical thickness",
            "Left Middle Temporal average cortical thickness",
            "Right Middle Temporal average cortical thickness",
            "Left Superior Temporal average cortical thickness",
            "Right Superior Temporal average cortical thickness",
            "Left Fusiform average cortical thickness",
            "Right Fusiform average cortical thickness",
            "Left Temporal Pole average cortical thickness",
            "Right Temporal Pole average cortical thickness",
        ],
    },
    # Mean thickness across posterior temporoparietal regions.
    "posterior_temporoparietal_thickness": {
        "operation": "mean",
        "variables": [
            "Left Posterior Cingulate average cortical thickness",
            "Right Posterior Cingulate average cortical thickness",
            "Left Precuneus average cortical thickness",
            "Right Precuneus average cortical thickness",
            "Left Inferior Parietal average cortical thickness",
            "Right Inferior Parietal average cortical thickness",
            "Left Superior Parietal average cortical thickness",
            "Right Superior Parietal average cortical thickness",
            "Left Supramarginal average cortical thickness",
            "Right Supramarginal average cortical thickness",
        ],
    },
}


def normalize_variable_name(variable_name: str) -> str:
    """Normalize a FreeSurfer variable name for case-insensitive exact lookup."""
    return re.sub(r"\s+", " ", variable_name.strip()).lower()


def parse_mri_text(text: object) -> dict[str, float]:
    """Parse a structural MRI text cell into normalized variable names and values."""
    if text is None:
        return {}
    if isinstance(text, float) and math.isnan(text):
        return {}

    parsed: dict[str, float] = {}
    for raw_line in str(text).splitlines():
        line = BULLET_RE.sub("", raw_line.strip())
        if ":" not in line:
            continue

        variable_name, value_text = line.split(":", 1)
        normalized_name = normalize_variable_name(variable_name)
        if not normalized_name:
            continue

        value_match = NUMERIC_RE.search(value_text)
        if value_match is None:
            continue

        try:
            value = float(value_match.group(0))
        except ValueError:
            continue

        if math.isfinite(value):
            parsed[normalized_name] = value
    return parsed


def get_value(parsed_dict: dict[str, float], variable_name: str) -> float:
    """Return a parsed value by case-insensitive exact variable name, else NaN."""
    value = parsed_dict.get(normalize_variable_name(variable_name), np.nan)
    if value is None or not math.isfinite(float(value)):
        return float("nan")
    return float(value)


def required_values(
    parsed_dict: dict[str, float], variable_names: Iterable[str]
) -> list[float] | None:
    values = [get_value(parsed_dict, variable_name) for variable_name in variable_names]
    if any(math.isnan(value) for value in values):
        return None
    return values


def calculate_composites_for_row(mri_text: object) -> dict[str, float]:
    """Calculate the eight AD MRI composites for one structural MRI text cell."""
    parsed = parse_mri_text(mri_text)
    icv = get_value(parsed, ICV_VARIABLE)
    valid_icv = not math.isnan(icv) and icv > 0

    composites: dict[str, float] = {}
    for composite_name, definition in COMPOSITE_DEFINITIONS.items():
        values = required_values(parsed, definition["variables"])
        if values is None:
            composites[composite_name] = float("nan")
            continue

        if definition["operation"] == "mean":
            composites[composite_name] = float(np.mean(values))
        elif definition["operation"] == "sum_icv":
            composites[composite_name] = float(np.sum(values) / icv) if valid_icv else float("nan")
        else:
            raise ValueError(f"Unknown composite operation: {definition['operation']}")

    return composites


def format_value(value: float) -> str:
    """Format values for the replacement MRI text."""
    if value is None or math.isnan(float(value)):
        return "NaN"
    return f"{float(value):.12g}"


def format_composites_text(composites: dict[str, float]) -> str:
    """Format the eight composites as the replacement structural MRI text."""
    lines = ["Structural MRI AD composites:"]
    for composite_name in COMPOSITE_DEFINITIONS:
        lines.append(
            f"- {composite_name}: {format_value(composites.get(composite_name, float('nan')))}"
        )
    return "\n".join(lines)
