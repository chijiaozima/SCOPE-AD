# ADNI baseline data for sequential clinical decision-making

This package reproduces the 4090pc experiment's 2,218 baseline episodes. The locked split contains **1,552 training, 333 validation and 333 test participants**. Each participant belongs to one split. The included `pseudo_dataset/` contains 60 independently generated fictional participants.

## Run the public example

From this directory, use Python 3.12 and install the CPU dependencies. The data and weights for the original study are unnecessary for this example.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python example.py
```

The example loads the included 42/9/9 split through the 4090pc episode parser. For the first synthetic test participant it shows the initially visible demographics, available actions, the result of one scripted cognitive assessment and the remaining actions. The example starts with a $500 budget and charges $50 for the assessment. The remaining budget is $450, and a second cognitive assessment is no longer available. The three terminal diagnosis actions remain available. The example stops after this request.

To regenerate the same three CSVs and their assignment file, choose a new directory. The generator uses hand-written fields and a fixed random seed, with no source-data input. The tiny dataset is for exercising the file format and episode workflow.

```bash
python make_pseudo_dataset.py --output /tmp/new-synthetic-cdm
```

## Reproduce the research cohort with ADNI access

### 1. Obtain the source tables

Use an approved ADNI account and keep the downloads in a private directory outside the Git checkout. The study used **All Subjects, all visits, download dated 28 May 2026**. If you have the authorized `Tables_28May2026.zip`, extract its six CSV files without opening and re-saving them in a spreadsheet. Obtain the matching `DATADIC.csv` separately. For a fresh export, follow these steps.

1. Sign in to the LONI IDA, select ADNI and open **Search and Download → ARC Builder**. Use **All Subjects**, across all available phases and visits. Leave diagnosis, visit, imaging and biomarker filters unset. The preparation script applies the baseline and availability rules after download.
2. Build the custom **My Table** using the field list and source-table guide below. Keep the ARC-generated `subject_id` and `visit` keys. Select variables by their exported field codes. Use the original 88-column selection below when recreating the view from scratch. An existing export can be reduced to the 67 required columns while preserving all its rows.
3. Add the five supporting tables below using the table-level selection in ARC Builder. In **Downloads → Tables**, download **My Table** and the five full tables for **All Subjects**. Preserve the exported rows, visit codes, column names and missing values. Save one copy of each table in the same private directory. The loader recognizes the ADNI download-date suffix. Mixing multiple dated copies of one table produces an error. [ADNI guide to ARC Builder](https://adni.loni.usc.edu/quick-start-guide-asset101625/arc_builder.html) · [ADNI instructions for downloading tables](https://adni.loni.usc.edu/support/experts-knowledge-base/question/?QID=1988)
4. Open **Study Files → Study Info → Data and Database**. Download **Data Dictionary [ADNI1,GO,2,3,4]**, extract `DATADIC.csv` and keep its original contents. The study used the dictionary for categorical labels and FreeSurfer feature names. [ADNI Study Files guide](https://adni.loni.usc.edu/quick-start-guide-asset/study_files.html)
5. Run the input check below before preparing the data. ADNI appends the download date to its filenames and updates tables over time. The reference identifies the inputs by content. If today's export differs, use a matching previously authorized archive or ask ADNI whether that version is available. A new-account download of the historical version has not been verified. [Data versions and download dates](https://adni.loni.usc.edu/quick-start-guide-asset/datacollection.html)

| Table in the export | Role |
|---|---|
| `All_Subjects_My_Table_28May2026.csv` | Custom subject/visit table with diagnosis, cognitive assessments, clinical history, APOE and fluid biomarkers |
| `All_Subjects_PTDEMOG_28May2026.csv` | Demographic fields |
| `All_Subjects_UCBERKELEY_AMY_6MM_28May2026.csv` | Amyloid PET summaries |
| `All_Subjects_UCBERKELEY_TAU_6MM_28May2026.csv` | Tau PET summaries |
| `All_Subjects_UCD_WMH_28May2026.csv` | Structural MRI summaries used in assembling the MRI record |
| `All_Subjects_UCSFFSX7_28May2026.csv` | FreeSurfer measurements used to calculate eight MRI composites |

For the custom `My_Table` export, select all subjects and visits and include the following fields. Additional columns are allowed. Keep `subject_id` and `visit` as the exported keys. The five supporting tables use `PTID` and visit fields `VISCODE2` / `VISCODE`. Export the complete supporting tables, including the FreeSurfer `ST*` measurements. The local dictionary supplies field-code meanings and FreeSurfer measurement names.

| Group | Required fields in My_Table |
|---|---|
| Episode and diagnosis | `subject_id, visit, DIAGNOSIS` |
| Cognitive and APOE | `MMSCORE, GDTOTAL, FAQTOTAL, NPITOTAL, TOTAL13, GENOTYPE` |
| Blood | `pT217_F, AB42_F, AB40_F, AB42_AB40_F, pT217_AB42_F, NfL_Q, GFAP_Q, pT217_C2N, npT217_C2N, AB42_C2N, AB40_C2N, AB42_AB40_C2N, pT217_npT217_C2N, APS2_C2N, APOE_C2N, NfL_F, GFAP_F` |
| CSF | `ABETA40, ABETA42, TAU, PTAU` |
| Family history | `MOTHDEM, MOTHAD, MOTHSXAGE, FATHDEM, FATHAD, FATHSXAGE, SIBDEMENT, SIBAD, SIBSXAGE, FHQMOM, FHQMOMAD, FHQDAD, FHQDADAD` |
| Other history | `IHSYMPTOM, IHPRESENT, IHCHRON, IHSEVER, IHONGOING, MHPSYCH, MH2NEURL, MH3HEAD, MH4CARD, MH5RESP, MH9ENDO, MH14ALCH, MH14AALCH, MH14BALCH, MH14CALCH, MH15DRUG, MH15ADRUG, MH15BDRUG, MH16SMOK, MH16ASMOK, MH16BSMOK, MH16CSMOK, HMSCORE, IHDESC` |

#### Finding the fields in ARC Builder

The archived CSV stores the exported field names. The archive contains no saved ARC query. The following table gives lookup locations from the study dictionary and ADNI's domain guides. Some field codes occur under several assays or study phases. The input check identifies which full ordered columns match the historical selection.

| ARC field group | Tables or forms to search |
|---|---|
| Clinical diagnosis | Diagnostic Summary (`DXSUM`), field `DIAGNOSIS`. |
| Cognitive scores | MMSE (`MMSCORE`), GDSCALE (`GDTOTAL`), FAQ (`FAQTOTAL`), NPI (`NPITOTAL`), ADAS-Cog (`TOTAL13`). For ADAS use the cross-phase data, including ADNI1 and GO/2/3 records. |
| APOE | Genetics Core APOE Results (`APOERES`), field `GENOTYPE`. |
| Family history | `FAMHXPAR` for parental history, `FAMHXSIB` for siblings, `FHQ` for the four `FHQ*` fields. |
| Other history | `INITHEALTH` for `IH*` (including `IHDESC`), `MEDHIST` for `MH*`, `MODHACH` for `HMSCORE`. |
| Plasma | Under Biospecimen Results use UPENN Plasma Biomarkers, Fujirebio & Quanterix (`UPENN_PLASMA_FUJIREBIO_QUANTERIX`) for the `_F` and `_Q` measurements. Look up `_C2N` fields under `C2N_PRECIVITYAD2_PLASMA`. The archived My Table includes `NfL_F` and `GFAP_F` even though the archived dictionary omits those two names. |
| CSF | Under Biospecimen Results, look up the four CSF measurements together under UPENN CSF Biomarkers Roche Elecsys (`UPENNBIOMK_ROCHE_ELECSYS`). Several assay tables reuse these names. Use the ordered column check to verify exported values. |

The source mapping for the non-unique names is a guide to locating the matching export. The 67 full-column checksums are the reproducibility test. [ADNI guide to biofluid and genotype tables](https://adni.loni.usc.edu/quick-start-guide-asset/bioflood.html)

The original custom export also included 21 auxiliary fields. Selecting them recreates the original column set and helps preserve the ARC visit population. These columns may be dropped **after export**. Demographic model inputs come from the separate PTDEMOG table.

| Source | Additional fields in the original My Table |
|---|---|
| PTDEMOG | `PTGENDER, PTRACCAT, PTETHCAT, PTNOTRT, PTWORK, PTWORKHS, PTEDUCAT, PTMARRY, PTDOBYY` |
| PHYSICAL | `PXGENAPP, PXHEADEY, PXHEART, PXPERIPH, PXABNORM` |
| VITALS | `VSWEIGHT, VSWTUNIT, VSHEIGHT, VSHTUNIT, VSBPSYS, VSBPDIA, VSPULSE` |

From this directory, in the data preparation environment, check the selection without writing participant data.

```bash
python prepare_adni.py --check-inputs \
  --raw-dir /private/study/Tables_28May2026 \
  --dictionary /private/study/DATADIC.csv
```

Expect **5 supporting tables, 1 dictionary, 67 required fields and 24,865 My Table rows**. The check compares the five complete files and dictionary by SHA-256. For My Table it compares each required column across all rows in export order, so extra fields and column ordering are immaterial. A differing field name in the error identifies which selection to revisit. If many fields differ, check the visit population, row order and export version. The reference contains aggregate digests only. [Frozen reference](reference.json)

### 2. Build the final split

Set the three paths to locations in your private workspace. Choose a new output directory. After dependency installation this command runs offline on CPU. Study files, real split CSVs and the participant assignment list belong in authorized storage.

```bash
python prepare_adni.py \
  --raw-dir /private/study/Tables_28May2026 \
  --dictionary /private/study/DATADIC.csv \
  --output /private/study/derived/4090pc-seed42
```

A successful run writes `train.csv`, `val.csv`, `test.csv`, `split_ids_seed42.csv` and `manifest.json` in the private output directory. The assignment file contains `patient_id,label,split`. The JSON records counts and file checksums. The command verifies the historical input shards, the final assignment and the exported CSVs with the project reader before making the output directory available.

| Final split | CN | MCI | AD | Total |
|---|---:|---:|---:|---:|
| Training | 659 | 672 | 221 | **1,552** |
| Validation | 141 | 144 | 48 | **333** |
| Test | 141 | 145 | 47 | **333** |

The expected SHA-256 of `split_ids_seed42.csv` is `b8fbd61d5ba4e4d3455958dd118108c44b4ac3d94f151f1cdb512c9fc5c528f7`. This checks every participant's identity, label and assignment. If the build reports a shard mismatch, compare your input filenames and checksums with [the verified snapshot](reference.json) and check the exported row order and dictionary version.

After this check, pass the **same private output directory** to the [research evaluation command](../eval/README.md). It rechecks all four CSVs before scoring the held-out test set.

### 3. How the split is constructed

1. Use the original My_Table row position as `row_id`. Select baseline (`bl`) rows with diagnosis codes 1, 2 or 3, mapped to CN, MCI or AD. There are 2,956 labelled baseline rows. For each modality, select the visit using the historical fallback rules, then prefer the candidate with the most observed fields. Resolve ties by original source row position.
2. Recreate the historical 70/20/10 label-stratified source groups with seed 42. Within each group remove a row when at least seven of the eight MRI composites are missing, or when both blood and CSF biomarkers are unavailable. The resulting input groups contain **1,536 / 464 / 218** participants. These groups determine the input order for step 3.
3. Concatenate those groups in train, validation, test order, preserving row order. Use scikit-learn to take a label-stratified 70% training partition with seed 42. Split the remainder equally with the same seed and stratification. Sort indices within each partition before writing. This gives the **1,552 / 333 / 333** research split. The source groups are reconstructed in memory.
4. Verify that each subject occurs once, all three partitions are disjoint, and the locked assignment matches the 4090pc digest. Read the final CSVs back through the episode parser to check labels, assignments, visible fields and modality availability. Future experiments can reuse this assignment directly.

## CSV schema and episode inputs

Each of the three CSVs has the same 13 columns, in order:

```text
row_id, subject_id, visit, cognitive_scales, demographic, medical_history,
apoe_genotype, blood_biomarkers, csf_biomarkers, structural_mri, amyloid_pet,
tau_pet, diagnose
```

Each modality cell contains a text header and lines of the form `- field: value`. A missing modality can use `<modality>: not available.`. Read these files with a CSV reader that supports quoted multiline fields. At episode start only demographics are visible. Cognitive assessment, APOE, blood, CSF, structural MRI, amyloid PET and tau PET become visible when requested. The loader keeps the diagnosis as a target and excludes free-text medical history from the model inputs. [Loader configuration](preparation/loader_config.json)

The raw-table transforms derive from the project's AD-Benchmark preparation scripts. The ordered final split and episode reader derive from the 4090pc ordinal and Soft EBM projects. The benchmark source notice is included in [LICENSE-AD-Benchmark.txt](LICENSE-AD-Benchmark.txt).
