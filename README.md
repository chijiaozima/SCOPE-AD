# SCOPE-AD

**Why spend so much money on a diagnosis when a cheaper one is available?**

Cost-aware sequential diagnosis on the CN–MCI–AD continuum.

This project reproduces the 4090pc workflow for choosing the next assessment or a terminal diagnosis (CN, MCI or AD) under a fixed budget. You can run the **trained research checkpoints** immediately or **train new teachers and students** from a small, independently generated dataset. Both public examples work without an ADNI account. Run the commands from the repository root.

| Directory | Contents |
|---|---|
| [data_process/](data_process/) | ADNI preparation, split verification and synthetic data |
| [train/teacher/](train/teacher/) | Ordinal diagnosis, five-fold cross-fitting, calibration and EBM teacher training |
| [train/distillation/](train/distillation/) | MLP and Qwen student distillation |
| [eval/](eval/) | Locked-test evaluation and per-seed, per-budget reference results |
| [checkpoints/](checkpoints/) | Research weights, configurations and inference example |

## Set up Python 3.12

The CPU workflow covers the ordinal model, the EBM and both MLP student branches. Create the environment and run directories alongside this project so that the project folder remains ready to share.

```bash
python3.12 -m venv ../cdm-env
source ../cdm-env/bin/activate
export PYTHONDONTWRITEBYTECODE=1
python -m pip install -r requirements.txt
```

On a Linux host with `uv`, `uv venv --seed --python python3.12 ../cdm-env` also creates the environment. Use the same activation and installation commands.

## A. Load the research checkpoints and run inference

```bash
python checkpoints/example.py --seed 42
```

This loads the final ordinal model, its fitted preprocessor and calibration, and the selected EBM, soft MLP and hard-action MLP. It uses the bundled synthetic validation case `SYN-0005`, initially observing only demographics with a $500 budget. The command prints the three diagnosis probabilities and a masked distribution over ten actions for each policy. For seed 42, the expected posterior is approximately `CN 0.0588, MCI 0.5897, AD 0.3516`. All three policies choose `REQUEST_COGNITIVE`. Seeds 123 and 2026 select the corresponding teacher and students. The final ordinal model always uses seed 42. [Checkpoint inventory](checkpoints/manifest.json) · [Recorded example output](checkpoints/example_result.json)

## B. Generate synthetic data, train teachers and distill students

Use new output paths. The paths below pass each stage's actual output directly to the next stage. Keep the teacher run at its original location because its manifest binds files by path and checksum.

```bash
python data_process/make_pseudo_dataset.py \
  --output ../cdm-runs/pseudo

python train/teacher/run_teacher.py --profile example \
  --data ../cdm-runs/pseudo \
  --output ../cdm-runs/teacher

python train/distillation/run_distillation.py --profile example --students mlp \
  --teacher-run ../cdm-runs/teacher \
  --output ../cdm-runs/students-mlp
```

1. The generator creates 60 fictional baseline cases and a fixed **42 / 9 / 9** train / validation / test split. The generated CSVs match the bundled fixture. To inspect one scripted acquisition on the bundled data, run `python data_process/example.py`.
2. Teacher training fits the final ordinal model, runs five-fold cross-fitting, calibrates the models, builds budget-specific targets and trains EBM teachers for seeds 42, 123 and 2026. The example uses one $500 budget and at most two epochs per model.
3. Distillation generates teacher trajectories from training and validation patients and trains one soft-policy MLP and one hard-action MLP per teacher seed for one epoch. The verified run produced **339 training, 73 validation and 0 test decision records**.

Read `../cdm-runs/teacher/example_result.json` and `../cdm-runs/students-mlp/example_result.json` for the checks on your newly trained models. Their predictions can differ from the research checkpoint output in A. The synthetic labels and measurements exercise the code and do not support a clinical performance estimate.

## Include the Qwen student on a GPU

Use Linux, CUDA-enabled PyTorch and sufficient GPU memory. The verified device was an RTX 4090. Install the extra dependencies in the active Python 3.12 environment. Download the pinned base model once if it is absent from your local Hugging Face cache. The checkpoint package includes the trained LoRA adapters and action heads.

```bash
python -m pip install -r requirements-qwen.txt
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
```

**Inference with all research policies** uses the same case as A.

```bash
python checkpoints/example.py --seed 42 --qwen
```

**Distillation with all three student branches** reuses the synthetic teachers produced in B. This command trains both MLP variants and the Qwen soft-policy student for each seed and checks the loaded seed-42 policies.

```bash
python train/distillation/run_distillation.py --profile example --students all \
  --teacher-run ../cdm-runs/teacher \
  --output ../cdm-runs/students-all
```

## C. Reproduce the real ADNI split and evaluate all four research policies

Follow the [ADNI table selection and export instructions](data_process/README.md#reproduce-the-research-cohort-with-adni-access). Use the original export or one that passes the frozen input checks. Set `STUDY` to a private directory **outside this repository**. Use one environment for reproducing the raw CSV build and the model environment from sections A and B for evaluation.

```bash
export STUDY=/private/study
python3.12 -m venv ../cdm-prep-env
../cdm-prep-env/bin/python -m pip install -r data_process/requirements.txt
../cdm-prep-env/bin/python data_process/prepare_adni.py --check-inputs \
  --raw-dir "$STUDY/Tables_28May2026" --dictionary "$STUDY/DATADIC.csv"
../cdm-prep-env/bin/python data_process/prepare_adni.py \
  --raw-dir "$STUDY/Tables_28May2026" --dictionary "$STUDY/DATADIC.csv" \
  --output "$STUDY/derived/4090pc-seed42"

source ../cdm-env/bin/activate
python -m pip install -r requirements-qwen.txt
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
HF_HUB_OFFLINE=1 python eval/evaluate.py \
  --data "$STUDY/derived/4090pc-seed42" --output "$STUDY/results/locked-test.csv"
```

Step 1 must report **1,552 train / 333 validation / 333 test** and assignment SHA-256 `b8fbd61d5ba4e4d3455958dd118108c44b4ac3d94f151f1cdb512c9fc5c528f7`. The training code uses only the 1,552 training participants for the five cross-fitting folds. The final command loads the released weights and scores all 333 held-out participants for four policies, three seeds and five budgets, checking every aggregate row against the 4090pc archive. Use Linux and CUDA for the full four-policy run. [Evaluation commands, metrics and results at $1,000](eval/README.md) · [All 60 historical reference rows](eval/reference_test.csv)

## Research data and source notices

Authorized ADNI users can reproduce the original **1,552 / 333 / 333** study split from the specified raw tables and data dictionary. Keep source tables, participant assignments, prepared episodes and research run directories in private storage. The public data files contain only synthetic cases. The included research checkpoint parameters and aggregate run records come from 4090pc.

For the authorized raw-table build, Step 1 lists the standalone dependency versions used to verify that export. The root requirements cover the public synthetic workflow and checkpoint inference.

The source notices accompanying the AD-Benchmark preparation code and the Qwen tokenizer are included with the project. [AD-Benchmark license](data_process/LICENSE-AD-Benchmark.txt) · [Qwen base-model license](LICENSE-Qwen.txt)

## Release checks

On 19 November 2026, both workflows ran from a separate copy of this project on 4090pc. Two fresh Python 3.12.3 environments were installed from the root requirements, one for the CPU examples and one with the additional Qwen dependencies. The GPU environment used the locally cached public Qwen base model at the pinned revision. The seed-42 research checkpoint output matched the supplied JSON. The end-to-end run regenerated the data, fit three new teachers, trained all nine students and reloaded each seed-42 student to check its action probabilities.

The four synthetic CSVs were regenerated from the synthetic generator and compared byte for byte, including a separate check of their JSON manifest. Their 60 identifiers all have the `SYN-` prefix. A scan of the release files, including the contents of the PyTorch archives, found zero ADNI participant IDs against the private list of 2,218 study participants or the ADNI ID format. No file matched a known raw table, private data split or participant assignment by SHA-256.

All 47 configuration and model files listed in the checkpoint manifest passed their SHA-256 checks. The 18 PyTorch files contain tensor dictionaries, and the three SafeTensors files contain LoRA weights. The release contains no symbolic links or run directories.

For the real-data release check, both the archived 88-column My Table and its 67-column projection reproduced all four frozen Step 1 files. On the original RTX 4090 the released policies matched all **60/60** historical test rows. A fresh CPU environment independently matched all 45 EBM/MLP rows. The additional distributed CSV is the aggregate test reference. [Locked-test verification](eval/README.md#release-verification-on-19-november-2026)
