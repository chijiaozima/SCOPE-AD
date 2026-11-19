# Step 2 · Train the ordinal diagnosis model and the EBM teacher

Starting from the patient splits produced in Step 1, this package fits an ordinal diagnosis model, calibrates its probabilities, and trains an energy-based model (EBM) to choose the next assessment or a terminal diagnosis. The public example runs all five stages on CPU with 60 fictional patients. [Step 1 dataset and preparation](../../data_process/README.md)

## Run the public example

From `train/teacher/`, run the following commands. Choose an output directory that does not yet exist. Python 3.12 was used on 4090pc.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_teacher.py --profile example --output ../../../cdm-runs/teacher
```

The default input is the included `../../data_process/pseudo_dataset/`. To use a separate copy, pass `--data /path/to/pseudo_dataset`. The runner checks the Step 1 manifest and SHA-256 digests of the three patient CSVs and the split assignment before training. It records each stage's output in `logs/`, the effective configuration in `configs/`, and the training products in `artifacts/` inside the new run directory.

The final check loads all three trained teachers and queries them on a validation patient with only demographics observed. It checks that each teacher returns probabilities over ten actions, that these sum to one, and that unavailable or unaffordable actions have zero probability. To repeat the check on your completed example, run `python verify_example.py ../../../cdm-runs/teacher`.

### Verified result on 4090pc (CPU)

| Check | Observed result |
|---|---|
| Patients in training / validation / test | 42 / 9 / 9 |
| Patients held out in successive training folds | 9 / 9 / 8 / 8 / 8. Each of the 42 patients appeared in exactly one held-out fold. That fold produced all predictions for that patient's subsets. |
| Calibration | The final model and all five fold models passed the example thresholds. Final validation Brier was 0.6668 and expected calibration error (ECE) was 0.00168. |
| Cached subsets and budget-specific targets | 2,880 patient/subset states. At the $500 budget, 218 training and 48 validation targets were used. All 40 test targets were excluded from fitting and selection. |
| Teacher policy | All three seeds produced a valid policy over ten actions. For synthetic patient `SYN-0005` at $500, seed 42 assigned the largest probability (0.584) to `REQUEST_COGNITIVE`. |

These figures describe a workflow test. The synthetic measurements are drawn without using the diagnosis label. A uniform three-class prediction has a multiclass Brier score of 2/3. The example uses a Brier ceiling of 0.80 to let training proceed. Its final score of 0.6668 exceeds the 0.60 research ceiling. The full-precision results and all three action distributions are in the [verified example output](example_result.json).

## Training sequence

| Stage | Entry point | What it fits or produces |
|---|---|---|
| 1. Ordinal diagnosis and final calibration | [train_ordinal.py](scripts/train_ordinal.py) | Fits preprocessing and the diagnosis model on training patients. Demographics and the observed subset of seven modalities predict probabilities for cognitively normal (CN), mild cognitive impairment (MCI), and Alzheimer's disease (AD), in that order. Validation macro-F1 selects the epoch. On partial validation states, the calibrator fits a positive temperature and two ordered thresholds by minimizing negative log-likelihood (NLL). |
| 2. Five-fold cross-fitting | [build_crossfit_beliefs.py](scripts/build_crossfit_beliefs.py) | Builds five stratified folds within the training patients. Each fold fits preprocessing on its outer fitting patients and uses an inner 90/10 split for model fitting and epoch selection. Each fold fits its own calibrator on the common, separate validation cohort. A training patient's probabilities come only from the fold that held that patient out. Validation and test probabilities come from the frozen final model in stage 1. |
| 3. Observed subset states | [cache_subset_states.py](scripts/cache_subset_states.py) | Enumerates available modality subsets for each patient. A common 64-dimensional encoding of visible features keeps the representation comparable across folds. Each state carries the calibrated probabilities and modality masks used by the planner. |
| 4. Offline action targets | [build_soft_bellman_targets.py](scripts/build_soft_bellman_targets.py) | Computes budget-specific action values backwards over subset transitions. A request incurs its acquisition cost and reveals the corresponding observed result in the next state. Terminal diagnosis costs come from the calibrated probabilities and ordinal risk. A patient can have several training targets, one for each feasible state and budget. |
| 5. EBM teacher | [train_ebm_planner.py](scripts/train_ebm_planner.py) | Fits action energies on training targets using Huber loss and an action-ranking KL loss. Lowest validation loss selects one model per seed. The model takes 87 state features and scores ten actions, seven requests and three terminal diagnoses. The training audit records the excluded test rows. |

The validation patients are reused for calibration and model selection. The reported validation calibration metrics are therefore in-sample for the calibrator. The test split stays separate throughout fitting, calibration and epoch selection.

## Research and example configurations

The original [data](configs/data.yaml), [ordinal](configs/ordinal.yaml), [cost](configs/costs.yaml), [experiment](configs/experiments.yaml) and [planner](configs/planner.yaml) configurations are included. The runner sets the input paths in the run copy. Only the example applies the [CPU overrides](example.yaml).

| Setting | 4090pc research run | Public example |
|---|---|---|
| Patient split (train / validation / test) | 1,552 / 333 / 333 | 42 / 9 / 9 |
| Device | RTX 4090, ordinal device auto, planner CUDA | CPU |
| Maximum epochs (ordinal / EBM) | 80 / 150 | 2 / 2 |
| Training budgets (USD) | 500, 1,000, 2,000, 4,000, 8,520 | 500 |
| Maximum Brier / maximum ECE | 0.60 / 0.10 | 0.80 / 0.10 |
| Allowed increase in calibration NLL | 1e-8 | 1e-8 |
| Cross-fitting / EBM seeds | 5 folds / 42, 123, 2026 | Same |
| Planner hidden widths / maximum acquisitions / energy temperature | 256–256–128 / 7 / 0.3 | Same |

### Run with an authorized ADNI dataset

First complete [Step 1](../../data_process/README.md) in private storage. Supply its output directory containing `train.csv`, `val.csv`, `test.csv`, `split_ids_seed42.csv` and `manifest.json`. Use a CUDA-enabled PyTorch installation for the research profile. The recorded 4090pc environment used Python 3.12.3 and PyTorch 2.13.0+cu130.

```bash
python run_teacher.py --profile research \
  --data /private/adni/step1-output \
  --output /private/adni/teacher-run
```

The research profile checks the locked seed-42 patient assignment against its recorded SHA-256 and requires the original 1,552 / 333 / 333 counts. Keep the entire run directory in authorized storage. Intermediate files include patient-level states, predictions and fold assignments.

## Historical provenance

The five entry points, their imported project modules, and the five configuration files came from the 4090pc project. The 9 August 2026 training record matches the data and ordinal configuration hashes, resolved planner settings, and canonical state encoder. The recorded best epoch was 22 for the final ordinal model and 14, 8 and 11 for EBM seeds 42, 123 and 2026, respectively. The final three entry points on the host were saved again on 2 September and include later experiment options. Their exact August source revision could not be confirmed. This package was rerun on the public pseudo dataset. The historical cohort counts and training records are summarized in the [research reference](research_reference.json).
