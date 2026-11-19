"""Rebuild the filtered input shards from six source tables and the ADNI data dictionary."""

import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import modalities as m
from . import readable
from .filtering import build_keep_mask
from .mri import calculate_composites_for_row, format_composites_text
from .source_partition import split_by_label


def locate_tables(raw_dir):
    """Accept the archived names or one dated All_Subjects export of each table."""
    paths = {}
    for key, filename in m.DEFAULT_FILES.items():
        prefix = filename.removesuffix("28May2026.csv")
        pattern = re.compile(re.escape(prefix) + r"\d{1,2}[A-Za-z]{3}\d{4}\.csv")
        matches = [p for p in raw_dir.glob(prefix + "*.csv") if pattern.fullmatch(p.name)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one dated CSV for {prefix.rstrip('_')}, found {len(matches)}"
            )
        paths[key] = matches[0]
    return paths


def my_table_fields():
    return [
        "subject_id",
        "visit",
        "DIAGNOSIS",
        *m.MEDICAL_HISTORY_FIELDS,
        *m.COGNITIVE_FIELDS,
        *m.APOE_FIELDS,
        *m.BLOOD_FIELDS,
        *m.CSF_FIELDS,
    ]


def ordered_column_hashes(path):
    """Hash each full ordered column. Length prefixes preserve empty and multiline cells."""
    fields = my_table_fields()
    digests = {field: hashlib.sha256() for field in fields}
    count = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        missing = [field for field in fields if columns.count(field) != 1]
        if missing:
            raise ValueError("Missing or duplicated My_Table fields: " + ", ".join(missing))
        for row in reader:
            count += 1
            for field, digest in digests.items():
                cell = row[field].encode("utf-8")
                digest.update(len(cell).to_bytes(8, "big"))
                digest.update(cell)
    return count, {field: digest.hexdigest() for field, digest in digests.items()}


def check_inputs(raw_dir, dictionary, reference):
    paths = locate_tables(raw_dir)
    raw_hashes = reference["input_file_sha256"]
    for key, path in [*paths.items(), ("dictionary", dictionary)]:
        if key == "my_table":
            continue
        expected_name = "DATADIC.csv" if key == "dictionary" else m.DEFAULT_FILES[key]
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != raw_hashes[expected_name]:
            raise ValueError(f"{key}: file bytes differ from the archived export")
    count, columns = ordered_column_hashes(paths["my_table"])
    expected = reference["my_table"]
    changed = [
        field
        for field, digest in columns.items()
        if expected["ordered_column_sha256"][field] != digest
    ]
    if count != expected["rows"] or changed:
        raise ValueError(
            f"My_Table: {count} rows; differing ordered columns: " + ", ".join(changed)
        )
    print(
        f"Verified five supporting tables, dictionary and {len(columns)} My_Table fields across {count} rows."
    )


def read_dictionary(path):
    """Use field definitions from the local IDA DATADIC export; patient data stay offline."""
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"TBLNAME", "FLDNAME", "CODE", "UNITS", "TEXT"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("DATADIC must contain " + ", ".join(sorted(required)))
        entries = list(reader)

    # The historical readability map combines fields with the same display name.
    # Its tie-break for conflicting descriptions is alphabetical raw field name.
    source_fields = defaultdict(list)
    for field in m.DEMOGRAPHIC_FIELDS + m.MEDICAL_HISTORY_FIELDS:
        source_fields[m.display_label(field)].append(field)
    by_field = defaultdict(list)
    for item in entries:
        by_field[item["FLDNAME"]].append(item)
    code_maps = {}
    for name, aliases in source_fields.items():
        codes = (row["CODE"] for field in sorted(aliases) for row in by_field[field])
        code_maps[name] = readable.parse_code_map(*codes)

    fs_fields = {}
    for table in ("UCSFFSX6", "UCSFFSX51", "UCSFFSX", "UCSFFSX7"):
        for item in entries:
            field = item["FLDNAME"]
            if (
                item["TBLNAME"] != table
                or field in fs_fields
                or not m.FSX_MEASURE_FIELD_RE.fullmatch(field)
            ):
                continue
            description = f"{item['UNITS']} {item['TEXT']}"
            fs_fields[field] = readable.parse_freesurfer_description(description)
    return code_maps, fs_fields


def baseline_rows(raw_dir: Path, dictionary: Path):
    code_maps, fs_labels = read_dictionary(dictionary)
    tables = {
        name: m.read_source_table(name, path) for name, path in locate_tables(raw_dir).items()
    }
    fs_fields = m.clean_freesurfer_measure_fields(tables["fsx"].columns)
    if not fs_fields:
        raise ValueError("UCSFFSX7 has no FreeSurfer ST measurement fields")
    required = {
        "my_table": my_table_fields(),
        "demographics": m.DEMOGRAPHIC_FIELDS,
        "amyloid_pet": m.CLEAN_AMYLOID_FIELDS,
        "tau_pet": m.CLEAN_TAU_FIELDS,
        "wmh": m.CLEAN_WMH_FIELDS,
        "fsx": fs_fields,
    }
    indexes = {}
    for name, table in tables.items():
        if name == "my_table":
            m.require_columns(table, required[name])
            anchors = m.add_keys(table.rows, "subject_id", ["visit"])
            indexes[name] = m.index_by_subject(anchors)
        else:
            m.require_columns(table, ["PTID", "VISCODE", "VISCODE2", *required[name]])
            indexes[name] = m.index_by_subject(
                m.add_keys(table.rows, "PTID", ["VISCODE2", "VISCODE"])
            )

    # Number before selecting baseline. The historical row_id is 1-based in My_Table.
    rows = []
    for row_id, anchor in enumerate(anchors, start=1):
        if anchor["_visit"] != "bl" or anchor["DIAGNOSIS"].strip() not in m.LABEL_MAP:
            continue
        subject = anchor["_subject_id"]

        def select(source, fields, fallbacks):
            return m.select_best_row(indexes[source], subject, "bl", fallbacks, fields)

        def clinical_text(title, source, fields, fallbacks):
            return m.render_modality(title, select(source, fields, fallbacks), fields, 80)

        structural = m.render_clean_structural_mri(
            select("wmh", m.CLEAN_WMH_FIELDS, ["scmri", "sc"]),
            select("fsx", fs_fields, ["scmri", "sc"]),
            fs_fields,
            fs_labels,
        )
        demographic = clinical_text("Demographic", "demographics", m.DEMOGRAPHIC_FIELDS, ["sc"])
        history = clinical_text("Medical history", "my_table", m.MEDICAL_HISTORY_FIELDS, ["sc"])
        rows.append(
            {
                "row_id": str(row_id),
                "subject_id": subject,
                "visit": "bl",
                "cognitive_scales": readable.light_normalize_modality(
                    clinical_text("Cognitive scales", "my_table", m.COGNITIVE_FIELDS, ["sc"])
                ),
                "demographic": readable.decode_demographic_or_history(demographic, code_maps),
                "medical_history": readable.decode_demographic_or_history(history, code_maps),
                "apoe_genotype": readable.light_normalize_modality(
                    clinical_text("APOE genotype", "my_table", m.APOE_FIELDS, ["sc"])
                ),
                "blood_biomarkers": readable.light_normalize_modality(
                    clinical_text("Blood biomarkers", "my_table", m.BLOOD_FIELDS, [])
                ),
                "csf_biomarkers": readable.light_normalize_modality(
                    clinical_text("CSF biomarkers", "my_table", m.CSF_FIELDS, [])
                ),
                "structural_mri": format_composites_text(calculate_composites_for_row(structural)),
                "amyloid_pet": m.render_clean_modality(
                    "Amyloid PET",
                    select("amyloid_pet", m.CLEAN_AMYLOID_FIELDS, []),
                    m.CLEAN_AMYLOID_FIELDS,
                ),
                "tau_pet": m.render_clean_modality(
                    "Tau PET", select("tau_pet", m.CLEAN_TAU_FIELDS, []), m.CLEAN_TAU_FIELDS
                ),
                "diagnose": m.render_diagnose(anchor),
            }
        )
    return rows


def rebuild_shards(raw_dir: Path, dictionary: Path):
    rows = baseline_rows(raw_dir, dictionary)
    if len({row["subject_id"] for row in rows}) != len(rows) or any(
        not row["subject_id"] for row in rows
    ):
        raise ValueError("Expected a unique subject ID for every labelled baseline visit")
    splits = split_by_label(rows, seed=42, train_frac=0.7, val_frac=0.2, test_frac=0.1)
    filtered = {}
    for name, entries in splits.items():
        frame = pd.DataFrame(entries, columns=m.OUTPUT_COLUMNS)
        keep, _, _ = build_keep_mask(
            frame, "structural_mri", "blood_biomarkers", "csf_biomarkers", 7
        )
        filtered[name] = frame.loc[keep].copy()
    return filtered
