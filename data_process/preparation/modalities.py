"""Source-table selection and field rendering from the ADNI-CDM preparation scripts."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .readable import FreeSurferField


DEFAULT_FILES = {
    "my_table": "All_Subjects_My_Table_28May2026.csv",
    "demographics": "All_Subjects_PTDEMOG_28May2026.csv",
    "amyloid_pet": "All_Subjects_UCBERKELEY_AMY_6MM_28May2026.csv",
    "tau_pet": "All_Subjects_UCBERKELEY_TAU_6MM_28May2026.csv",
    "wmh": "All_Subjects_UCD_WMH_28May2026.csv",
    "fsx": "All_Subjects_UCSFFSX7_28May2026.csv",
}


OUTPUT_COLUMNS = [
    "row_id",
    "subject_id",
    "visit",
    "cognitive_scales",
    "demographic",
    "medical_history",
    "apoe_genotype",
    "blood_biomarkers",
    "csf_biomarkers",
    "structural_mri",
    "amyloid_pet",
    "tau_pet",
    "diagnose",
]


MISSING_VALUES = {"", " ", "na", "n/a", "nan", "none", "null", ".", "-4", "-1"}


LABEL_MAP = {"1": "CN", "2": "MCI", "3": "AD"}


DEMOGRAPHIC_FIELDS = [
    "PTGENDER",
    "PTEDUCAT",
    "PTMARRY",
    "PTETHCAT",
    "PTRACCAT",
    "PTNOTRT",
    "PTWORK",
    "PTWORKHS",
    "PTDOBYY",
]


MEDICAL_HISTORY_FIELDS = [
    "MOTHDEM",
    "MOTHAD",
    "MOTHSXAGE",
    "FATHDEM",
    "FATHAD",
    "FATHSXAGE",
    "SIBDEMENT",
    "SIBAD",
    "SIBSXAGE",
    "FHQMOM",
    "FHQMOMAD",
    "FHQDAD",
    "FHQDADAD",
    "IHSYMPTOM",
    "IHPRESENT",
    "IHCHRON",
    "IHSEVER",
    "IHONGOING",
    "MHPSYCH",
    "MH2NEURL",
    "MH3HEAD",
    "MH4CARD",
    "MH5RESP",
    "MH9ENDO",
    "MH14ALCH",
    "MH14AALCH",
    "MH14BALCH",
    "MH14CALCH",
    "MH15DRUG",
    "MH15ADRUG",
    "MH15BDRUG",
    "MH16SMOK",
    "MH16ASMOK",
    "MH16BSMOK",
    "MH16CSMOK",
    "HMSCORE",
    "IHDESC",
]


COGNITIVE_FIELDS = ["MMSCORE", "GDTOTAL", "FAQTOTAL", "NPITOTAL", "TOTAL13"]


APOE_FIELDS = ["GENOTYPE"]


BLOOD_FIELDS = [
    "pT217_F",
    "AB42_F",
    "AB40_F",
    "AB42_AB40_F",
    "pT217_AB42_F",
    "NfL_Q",
    "GFAP_Q",
    "pT217_C2N",
    "npT217_C2N",
    "AB42_C2N",
    "AB40_C2N",
    "AB42_AB40_C2N",
    "pT217_npT217_C2N",
    "APS2_C2N",
    "APOE_C2N",
    "NfL_F",
    "GFAP_F",
]


CSF_FIELDS = ["ABETA40", "ABETA42", "TAU", "PTAU"]


WMH_FIELDS = [
    "MANUFACTURER",
    "MANUFACTURERSMODELNAME",
    "MAGNETICFIELDSTRENGTH",
    "MRACQUISITIONTYPEFLAIR",
    "CEREBRUM_TCV",
    "CEREBRUM_TCB",
    "CEREBRUM_TCC",
    "CEREBRUM_GRAY",
    "CEREBRUM_WHITE",
    "LEFT_HIPPO",
    "RIGHT_HIPPO",
    "TOTAL_HIPPO",
    "TOTAL_CSF",
    "TOTAL_GRAY",
    "TOTAL_WHITE",
    "TOTAL_WMH",
    "TOTAL_BRAIN",
]


TECHNICAL_WMH_FIELDS = {
    "MANUFACTURER",
    "MANUFACTURERSMODELNAME",
    "MAGNETICFIELDSTRENGTH",
    "MRACQUISITIONTYPEFLAIR",
}


CLEAN_WMH_FIELDS = [field for field in WMH_FIELDS if field not in TECHNICAL_WMH_FIELDS]


FSX_MEASURE_FIELD_RE = re.compile(r"^ST\d+(?:CV|SA|TA|TS|SV|HS)?$", re.IGNORECASE)


AMYLOID_FIELDS = [
    "TRACER",
    "TRACER_SUVR_WARNING",
    "AMYLOID_STATUS",
    "AMYLOID_STATUS_COMPOSITE_REF",
    "CENTILOIDS",
    "SUMMARY_SUVR",
    "SUMMARY_VOLUME",
    "WHOLECEREBELLUM_SUVR",
    "COMPOSITE_REF_SUVR",
    "CEREBELLUM_CORTEX_SUVR",
    "ERODED_SUBCORTICALWM_SUVR",
    "CTX_ENTORHINAL_SUVR",
    "CTX_PRECUNEUS_SUVR",
    "CTX_POSTERIORCINGULATE_SUVR",
    "HIPPOCAMPUS_SUVR",
]


CLEAN_AMYLOID_FIELDS = [
    field
    for field in AMYLOID_FIELDS
    if field not in {"TRACER", "TRACER_SUVR_WARNING", "SUMMARY_VOLUME"}
]


TAU_FIELDS = [
    "TRACER",
    "TRACER_SUVR_WARNING",
    "META_TEMPORAL_SUVR",
    "META_TEMPORAL_VOLUME",
    "CTX_ENTORHINAL_SUVR",
    "CTX_ENTORHINAL_VOLUME",
    "INFERIORCEREBELLUM_SUVR",
    "ERODED_SUBCORTICALWM_SUVR",
    "CTX_PRECUNEUS_SUVR",
    "CTX_POSTERIORCINGULATE_SUVR",
    "HIPPOCAMPUS_SUVR",
]


CLEAN_TAU_FIELDS = [
    field
    for field in TAU_FIELDS
    if field
    not in {
        "TRACER",
        "TRACER_SUVR_WARNING",
        "META_TEMPORAL_VOLUME",
        "CTX_ENTORHINAL_VOLUME",
    }
]


DISPLAY_LABELS = {
    "PTGENDER": "sex",
    "PTEDUCAT": "education_years",
    "PTMARRY": "marital_status",
    "PTETHCAT": "ethnicity",
    "PTRACCAT": "race",
    "PTNOTRT": "retirement_status",
    "PTWORK": "occupation",
    "PTWORKHS": "work_history_score",
    "PTDOBYY": "birth_year",
    "MOTHDEM": "mother_dementia_history",
    "MOTHAD": "mother_AD_history",
    "MOTHSXAGE": "mother_symptom_age",
    "FATHDEM": "father_dementia_history",
    "FATHAD": "father_AD_history",
    "FATHSXAGE": "father_symptom_age",
    "SIBDEMENT": "sibling_dementia_history",
    "SIBAD": "sibling_AD_history",
    "SIBSXAGE": "sibling_symptom_age",
    "FHQMOM": "mother_history",
    "FHQMOMAD": "mother_AD_history",
    "FHQDAD": "father_history",
    "FHQDADAD": "father_AD_history",
    "IHSYMPTOM": "initial_symptom",
    "IHPRESENT": "symptom_present",
    "IHCHRON": "chronicity",
    "IHSEVER": "severity",
    "IHONGOING": "ongoing_symptom",
    "MHPSYCH": "psychiatric_history",
    "MH2NEURL": "neurologic_history",
    "MH3HEAD": "head_injury_history",
    "MH4CARD": "cardiac_history",
    "MH5RESP": "respiratory_history",
    "MH9ENDO": "endocrine_history",
    "MH14ALCH": "alcohol_history",
    "MH14AALCH": "alcohol_current",
    "MH14BALCH": "alcohol_frequency",
    "MH14CALCH": "alcohol_quantity",
    "MH15DRUG": "drug_history",
    "MH15ADRUG": "drug_current",
    "MH15BDRUG": "drug_type",
    "MH16SMOK": "smoking_history",
    "MH16ASMOK": "smoking_current",
    "MH16BSMOK": "smoking_pack_years",
    "MH16CSMOK": "smoking_years_since_quit",
    "HMSCORE": "Hachinski_score",
    "IHDESC": "history_description",
    "MMSCORE": "MMSE",
    "GDTOTAL": "GDS_total",
    "FAQTOTAL": "FAQ_total",
    "NPITOTAL": "NPI_total",
    "TOTAL13": "ADAS_Cog_13_total",
    "GENOTYPE": "APOE_genotype",
    "ABETA40": "CSF_Abeta40",
    "ABETA42": "CSF_Abeta42",
    "TAU": "CSF_total_tau",
    "PTAU": "CSF_phosphorylated_tau",
    "CENTILOIDS": "amyloid_centiloids",
    "SUMMARY_SUVR": "amyloid_summary_SUVR",
    "META_TEMPORAL_SUVR": "tau_meta_temporal_SUVR",
    "TOTAL_HIPPO": "total_hippocampal_volume",
    "TOTAL_WMH": "total_white_matter_hyperintensity",
}


CLEAN_WMH_LABELS = {
    "CEREBRUM_TCV": ("Total cerebrum cranial volume, supratentorial portion", "cm3"),
    "CEREBRUM_TCB": ("Total cerebrum brain volume", "cm3"),
    "CEREBRUM_TCC": ("Total cerebrum CSF volume", "cm3"),
    "CEREBRUM_GRAY": ("Total cerebrum gray matter volume", "cm3"),
    "CEREBRUM_WHITE": ("Total cerebrum white matter volume", "cm3"),
    "LEFT_HIPPO": ("Segmented left hippocampus volume", "cm3"),
    "RIGHT_HIPPO": ("Segmented right hippocampus volume", "cm3"),
    "TOTAL_HIPPO": ("Total hippocampal volume", "cm3"),
    "TOTAL_CSF": ("Total CSF volume", "cm3"),
    "TOTAL_GRAY": ("Total gray matter volume", "cm3"),
    "TOTAL_WHITE": ("Total white matter volume", "cm3"),
    "TOTAL_WMH": ("Total white matter hyperintensity volume", "cm3"),
    "TOTAL_BRAIN": ("Total brain volume", "cm3"),
}


CLEAN_PET_LABELS = {
    "AMYLOID_STATUS": "cross-sectional amyloid PET status, cerebellum reference",
    "AMYLOID_STATUS_COMPOSITE_REF": "amyloid PET status, composite reference",
    "CENTILOIDS": "amyloid centiloids",
    "SUMMARY_SUVR": "amyloid summary SUVR",
    "WHOLECEREBELLUM_SUVR": "whole cerebellum SUVR",
    "COMPOSITE_REF_SUVR": "composite reference SUVR",
    "CEREBELLUM_CORTEX_SUVR": "cerebellum cortex SUVR",
    "ERODED_SUBCORTICALWM_SUVR": "eroded subcortical white matter SUVR",
    "CTX_ENTORHINAL_SUVR": "entorhinal cortex SUVR",
    "CTX_PRECUNEUS_SUVR": "precuneus cortex SUVR",
    "CTX_POSTERIORCINGULATE_SUVR": "posterior cingulate cortex SUVR",
    "HIPPOCAMPUS_SUVR": "hippocampus SUVR",
    "META_TEMPORAL_SUVR": "tau meta-temporal SUVR",
    "INFERIORCEREBELLUM_SUVR": "inferior cerebellum SUVR",
}


CLEAN_PET_VALUE_MAPS = {
    "AMYLOID_STATUS": {"0": "amyloid negative", "1": "amyloid positive"},
    "AMYLOID_STATUS_COMPOSITE_REF": {"0": "amyloid negative", "1": "amyloid positive"},
}


CATEGORY_MAPS = {
    "PTGENDER": {"1": "Male", "2": "Female"},
    "PTMARRY": {
        "1": "Married",
        "2": "Widowed",
        "3": "Divorced",
        "4": "Never married",
        "5": "Unknown",
    },
    "PTETHCAT": {"1": "Hispanic/Latino", "2": "Not Hispanic/Latino", "3": "Unknown"},
    "PTNOTRT": {"0": "No", "1": "Yes"},
}


@dataclass(frozen=True)
class SourceTable:
    name: str
    path: Path
    rows: list[dict[str, str]]
    columns: list[str]


@dataclass(frozen=True)
class Selection:
    row: dict[str, str] | None
    source_visit: str | None
    n_fields: int


def clean_scalar(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in MISSING_VALUES:
        return None
    return text


def normalize_visit(value: Any) -> str:
    return str(value or "").strip().lower()


def normalize_ptid(value: Any) -> str:
    return str(value or "").strip()


def read_source_table(name: str, path: Path) -> SourceTable:
    if not path.exists():
        raise FileNotFoundError(f"Required source file is missing for {name}: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    if not columns:
        raise ValueError(f"Source file has no header for {name}: {path}")
    return SourceTable(name=name, path=path, rows=rows, columns=columns)


def require_columns(table: SourceTable, required: list[str]) -> None:
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(
            f"{table.name} is missing required columns: {missing}. " f"File: {table.path}"
        )


def add_keys(
    rows: list[dict[str, str]],
    subject_column: str,
    visit_columns: list[str],
) -> list[dict[str, str]]:
    keyed = []
    for index, row in enumerate(rows):
        visit_value = ""
        for column in visit_columns:
            visit_value = normalize_visit(row.get(column))
            if visit_value:
                break

        item = dict(row)
        item["_source_row_index"] = str(index)
        item["_subject_id"] = normalize_ptid(row.get(subject_column))
        item["_visit"] = visit_value
        keyed.append(item)
    return keyed


def index_by_subject(rows: list[dict[str, str]]) -> dict[str, dict[str, list[dict[str, str]]]]:
    index: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        subject_id = row.get("_subject_id", "")
        visit = row.get("_visit", "")
        if subject_id and visit:
            index[subject_id][visit].append(row)
    return index


def unique_visits(anchor_visit: str, fallback_visits: list[str]) -> list[str]:
    visits = []
    for visit in [anchor_visit, *fallback_visits]:
        normalized = normalize_visit(visit)
        if normalized and normalized not in visits:
            visits.append(normalized)
    return visits


def count_nonmissing(row: dict[str, str], fields: list[str]) -> int:
    return sum(clean_scalar(row.get(field)) is not None for field in fields)


def select_best_row(
    index: dict[str, dict[str, list[dict[str, str]]]],
    subject_id: str,
    anchor_visit: str,
    fallback_visits: list[str],
    fields: list[str],
) -> Selection:
    visits = unique_visits(anchor_visit, fallback_visits)
    subject_rows = index.get(subject_id, {})
    for visit in visits:
        candidates = subject_rows.get(visit, [])
        if not candidates:
            continue

        ranked = sorted(
            candidates,
            key=lambda row: (-count_nonmissing(row, fields), int(row["_source_row_index"])),
        )
        best = ranked[0]
        n_fields = count_nonmissing(best, fields)
        if n_fields > 0:
            return Selection(row=best, source_visit=visit, n_fields=n_fields)
    return Selection(row=None, source_visit=None, n_fields=0)


def display_label(field: str) -> str:
    return DISPLAY_LABELS.get(field, field)


def display_value(field: str, value: Any) -> str:
    text = clean_scalar(value)
    if text is None:
        return ""
    return CATEGORY_MAPS.get(field, {}).get(text, text)


def values_from_row(row: dict[str, str] | None, fields: list[str]) -> list[tuple[str, str]]:
    if row is None:
        return []
    values = []
    for field in fields:
        value = display_value(field, row.get(field))
        if value:
            values.append((field, value))
    return values


def format_value_with_unit(value: str, units: str) -> str:
    if not units:
        return value
    normalized_value = value.strip().lower()
    normalized_units = units.strip().lower()
    if normalized_value.endswith(normalized_units):
        return value
    return f"{value} {units}"


def clean_pet_label(field: str) -> str:
    return CLEAN_PET_LABELS.get(field, display_label(field).replace("_", " "))


def clean_pet_value(field: str, value: str) -> str:
    return CLEAN_PET_VALUE_MAPS.get(field, {}).get(value, value)


def clean_freesurfer_measure_fields(columns: list[str]) -> list[str]:
    return sorted(
        (field for field in columns if FSX_MEASURE_FIELD_RE.fullmatch(field)),
        key=lambda field: field.upper(),
    )


def render_modality(
    title: str,
    selection: Selection,
    fields: list[str],
    max_text_fields: int,
) -> str:
    values = values_from_row(selection.row, fields)
    if not values:
        return f"{title}: not available."

    lines = [f"{title} source_visit={selection.source_visit}:"]
    for field, value in values[:max_text_fields]:
        lines.append(f"- {display_label(field)}: {value}")
    if len(values) > max_text_fields:
        lines.append(f"- {len(values) - max_text_fields} additional fields omitted from text.")
    return "\n".join(lines)


def render_clean_modality(title: str, selection: Selection, fields: list[str]) -> str:
    values = values_from_row(selection.row, fields)
    if not values:
        return f"{title}: not available."

    lines = [f"{title} source_visit={selection.source_visit}:"]
    for field, value in values:
        lines.append(f"- {clean_pet_label(field)}: {clean_pet_value(field, value)}")
    return "\n".join(lines)


def render_clean_structural_mri(
    wmh_selection: Selection,
    fsx_selection: Selection,
    fsx_fields: list[str],
    freesurfer_fields: dict[str, FreeSurferField],
) -> str:
    wmh_values = values_from_row(wmh_selection.row, CLEAN_WMH_FIELDS)
    fsx_values = values_from_row(fsx_selection.row, fsx_fields)
    if not wmh_values and not fsx_values:
        return "Structural MRI: not available."

    source_parts = []
    if wmh_values:
        source_parts.append(f"WMH={wmh_selection.source_visit}")
    if fsx_values:
        source_parts.append(f"FSX={fsx_selection.source_visit}")

    lines = [f"Structural MRI source_visit={', '.join(source_parts)}:"]
    for field, value in wmh_values:
        label, units = CLEAN_WMH_LABELS[field]
        lines.append(f"- {label}: {format_value_with_unit(value, units)}")
    for field, value in fsx_values:
        metadata = freesurfer_fields.get(field.upper())
        if metadata is None:
            raise RuntimeError(f"Missing readable FreeSurfer label for clean field: {field}")
        lines.append(f"- {metadata.readable_name}: {format_value_with_unit(value, metadata.units)}")
    return "\n".join(lines)


def render_diagnose(anchor_row: dict[str, str]) -> str:
    raw = clean_scalar(anchor_row.get("DIAGNOSIS"))
    if raw is None:
        return "Diagnose source_visit=current:\n- raw_DIAGNOSIS: not available\n- mapped_label: not available"
    mapped = LABEL_MAP.get(raw, "unmapped")
    return f"Diagnose source_visit=current:\n- raw_DIAGNOSIS: {raw}\n- mapped_label: {mapped}"
