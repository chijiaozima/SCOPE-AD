"""Decode ADNI field codes and format modality text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


MISSING_VALUES = {"", "na", "n/a", "nan", "none", "null", ".", "-4", "-1"}


@dataclass(frozen=True)
class FreeSurferField:
    units: str
    readable_name: str


def clean_scalar(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in MISSING_VALUES else text


def split_code_entries(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[|;]", text or "") if part.strip()]


def parse_code_map(*texts: str) -> dict[str, str]:
    code_map: dict[str, str] = {}
    for text in texts:
        for part in split_code_entries(text):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                continue
            if re.fullmatch(r"-?\d+(?:\.\d+)?\.\.-?\d+(?:\.\d+)?", part):
                continue
            code_map.setdefault(key, value)
    return code_map


def split_camel(text: str) -> str:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    replacements = {
        "Icv": "ICV",
        "Csf": "CSF",
        "WM": "white matter",
        "Hypo Intensities": "hypointensities",
        "Accumbens Area": "accumbens area",
        "Ventral DC": "ventral DC",
        "Bankssts": "banks of superior temporal sulcus",
        "Supra Marginal": "Supramarginal",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def parse_freesurfer_description(description: str) -> FreeSurferField:
    match = re.match(
        r"(?P<units>mm3|mm2|mm)\s+(?P<measure>.+?)\s+of\s+(?P<region>.+)$", description
    )
    if not match:
        return FreeSurferField("", description)

    units = match.group("units")
    measure = match.group("measure").strip()
    region = split_camel(match.group("region").strip())

    measure_labels = {
        "Cortical Volume (aparc.stats)": "cortical volume",
        "Subcortical Volume (aseg.stats)": "subcortical volume",
        "Surface Area (aparc.stats)": "surface area",
        "Thickness Average (aparc.stats)": "average cortical thickness",
        "Thickness Stardard Deviation (aparc.stats)": "cortical thickness standard deviation",
        "Thickness Standard Deviation (aparc.stats)": "cortical thickness standard deviation",
        "Volume (Cortical Parcellation)": "cortical parcellation volume",
        "Volume (WM Parcellation)": "white matter parcellation volume",
        "Surface Area": "surface area",
        "Cortical Thickness Average": "average cortical thickness",
        "Cortical Thickness Standard Deviation": "cortical thickness standard deviation",
    }
    measure_label = measure_labels.get(
        measure, measure[:1].lower() + measure[1:] if measure else ""
    )
    readable_name = f"{region} {measure_label}".strip()
    if region == "ICV" and measure_label.endswith("volume"):
        readable_name = "ICV volume"
    return FreeSurferField(units, readable_name)


TEXT_ALIASES = {
    "ethnicity": {
        "Hispanic/Latino": "Hispanic or Latino",
        "Not Hispanic/Latino": "Not Hispanic or Latino",
    }
}


def decode_code_value(
    field: str, value: str, code_maps: dict[str, dict[str, str]]
) -> tuple[str, str]:
    value = "" if value is None else str(value).strip()
    code_map = code_maps.get(field, {})
    if value in code_map:
        return code_map[value], "decoded_code"

    value = clean_scalar(value)
    if not value:
        return value, "missing"

    if field == "birth_year" and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value[:4], "year_normalized"

    alias = TEXT_ALIASES.get(field, {}).get(value)
    if alias is not None:
        return alias, "text_alias"

    if value in set(code_map.values()):
        return value, "already_text"

    if "|" in value and code_map:
        parts = value.split("|")
        if all(part in code_map for part in parts):
            return " + ".join(code_map[part] for part in parts), "decoded_multicode"

    return value, "unchanged"


def parse_bullet(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped.startswith("- ") or ":" not in stripped:
        return None
    field, value = stripped[2:].split(":", 1)
    return field.strip(), value.strip()


def rewrite_bullet_lines(
    text: str,
    rewrite: Callable[[str, str], str],
) -> str:
    if "not available" in text.lower() and ":" in text and "\n" not in text.strip():
        return text

    lines = []
    for line in text.splitlines():
        parsed = parse_bullet(line)
        if parsed is None:
            lines.append(line)
            continue
        field, value = parsed
        lines.append(rewrite(field, value))
    return "\n".join(lines)


def decode_demographic_or_history(
    text: str,
    code_maps: dict[str, dict[str, str]],
) -> str:
    def rewrite(field: str, value: str) -> str:
        decoded, _ = decode_code_value(field, value, code_maps)
        return f"- {field}: {decoded}"

    return rewrite_bullet_lines(text, rewrite)


LIGHT_FIELD_LABELS = {
    "MMSE": "MMSE",
    "GDS_total": "GDS total",
    "FAQ_total": "FAQ total",
    "NPI_total": "NPI total",
    "ADAS_Cog_13_total": "ADAS-Cog 13 total",
    "APOE_genotype": "APOE genotype",
    "pT217_F": "Fujirebio plasma p-tau217 (pT217_F)",
    "AB42_F": "Fujirebio plasma amyloid beta 42 (AB42_F)",
    "AB40_F": "Fujirebio plasma amyloid beta 40 (AB40_F)",
    "AB42_AB40_F": "Fujirebio plasma amyloid beta 42/40 ratio (AB42_AB40_F)",
    "pT217_AB42_F": "Fujirebio plasma p-tau217/amyloid beta 42 ratio (pT217_AB42_F)",
    "NfL_Q": "Quanterix plasma neurofilament light (NfL_Q)",
    "GFAP_Q": "Quanterix plasma GFAP (GFAP_Q)",
    "pT217_C2N": "C2N plasma p-tau217",
    "npT217_C2N": "C2N plasma non-phosphorylated tau217",
    "AB42_C2N": "C2N plasma amyloid beta 42",
    "AB40_C2N": "C2N plasma amyloid beta 40",
    "AB42_AB40_C2N": "C2N plasma amyloid beta 42/40 ratio",
    "pT217_npT217_C2N": "C2N plasma p-tau217/non-phosphorylated tau217 ratio",
    "APS2_C2N": "C2N APS2 score",
    "APOE_C2N": "C2N APOE result",
    "NfL_F": "Fujirebio plasma neurofilament light (NfL_F)",
    "GFAP_F": "Fujirebio plasma GFAP (GFAP_F)",
    "CSF_Abeta40": "CSF amyloid beta 40",
    "CSF_Abeta42": "CSF amyloid beta 42",
    "CSF_total_tau": "CSF total tau",
    "CSF_phosphorylated_tau": "CSF phosphorylated tau",
    "TRACER": "tracer",
    "TRACER_SUVR_WARNING": "SUVR warning",
    "AMYLOID_STATUS": "cross-sectional amyloid PET status, cerebellum reference",
    "AMYLOID_STATUS_COMPOSITE_REF": "amyloid PET status, composite reference",
    "amyloid_centiloids": "amyloid centiloids",
    "amyloid_summary_SUVR": "amyloid summary SUVR",
    "SUMMARY_VOLUME": "summary volume",
    "WHOLECEREBELLUM_SUVR": "whole cerebellum SUVR",
    "COMPOSITE_REF_SUVR": "composite reference SUVR",
    "CEREBELLUM_CORTEX_SUVR": "cerebellum cortex SUVR",
    "ERODED_SUBCORTICALWM_SUVR": "eroded subcortical white matter SUVR",
    "CTX_ENTORHINAL_SUVR": "entorhinal cortex SUVR",
    "CTX_ENTORHINAL_VOLUME": "entorhinal cortex volume",
    "CTX_PRECUNEUS_SUVR": "precuneus cortex SUVR",
    "CTX_POSTERIORCINGULATE_SUVR": "posterior cingulate cortex SUVR",
    "HIPPOCAMPUS_SUVR": "hippocampus SUVR",
    "tau_meta_temporal_SUVR": "tau meta-temporal SUVR",
    "META_TEMPORAL_VOLUME": "meta-temporal volume",
    "INFERIORCEREBELLUM_SUVR": "inferior cerebellum SUVR",
}


LIGHT_VALUE_MAPS = {
    "AMYLOID_STATUS": {"0": "amyloid negative", "1": "amyloid positive"},
    "AMYLOID_STATUS_COMPOSITE_REF": {"0": "amyloid negative", "1": "amyloid positive"},
}


def generic_readable_label(field: str) -> str:
    label = field.replace("_", " ")
    replacements = {
        "SUVR": "SUVR",
        "CSF": "CSF",
        "APOE": "APOE",
        "GFAP": "GFAP",
        "NfL": "NfL",
        "Abeta": "amyloid beta",
        "CTX": "cortex",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return " ".join(label.split())


def light_normalize_modality(text: str) -> str:
    def rewrite(field: str, value: str) -> str:
        label = LIGHT_FIELD_LABELS.get(field, generic_readable_label(field))
        decoded_value = LIGHT_VALUE_MAPS.get(field, {}).get(value, value)
        return f"- {label}: {decoded_value}"

    return rewrite_bullet_lines(text, rewrite)
