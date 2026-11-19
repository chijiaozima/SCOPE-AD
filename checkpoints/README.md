# Step 4 · 4090pc research checkpoints

This package contains the selected weights from the **August 2026 4090pc research run**. It includes the calibrated ordinal diagnosis model, five out-of-fold ordinal models, the EBM teachers, and three student branches. The example loads these study weights and runs inference on fictional input from [Step 1](../data_process/README.md). It requires no training or ADNI account.

## Which checkpoint to load

For inference, load the **final ordinal model, seed 42**, for every policy seed. To compare a student with its teacher, select the same seed for both. The table gives the validation-selected epoch in each seed column. The [manifest](manifest.json) maps each checkpoint to its directory, configuration, original artifact and SHA-256. It also lists the SHA-256 of every shipped configuration and model file.

| Model and weights | Training configuration | Seed 42 | Seed 123 | Seed 2026 |
|---|---|---:|---:|---:|
| [Final ordinal diagnosis model](weights/ordinal/) | [Ordinal](configs/ordinal.yaml), [data](configs/data.yaml) | 22 | — | — |
| [Soft EBM teacher](weights/ebm/) | [Planner](configs/planner.yaml), [costs](configs/costs.yaml) | 14 | 8 | 11 |
| [MLP student, soft KL](weights/mlp/) | [Distillation](configs/distillation.yaml), MLP and distillation settings | 100 | 48 | 100 |
| [Reactive MLP, hard-action baseline](weights/reactive_mlp/) | Same MLP configuration, trained on the teacher's chosen action | 6 | 54 | 21 |
| [Qwen3.5-9B student, soft KL](weights/qwen/) | [Distillation](configs/distillation.yaml), Qwen and LoRA settings | 1 | 3 | 2 |

The reactive MLP uses the teacher's seed as its run identifier. The 4090pc training script initializes this baseline after fitting the soft MLP in that seed's run and shuffles its training batches with `seed + 100003`. Only the soft-KL Qwen branch completed. The separately attempted hard-action Qwen run exited with code 139. [Teacher training](../train/teacher/README.md) · [Student distillation and run records](../train/distillation/README.md)

The [five fold models](weights/ordinal_oof/) generated held-out predictions for training patients during Step 2. Each has its own encoder and calibration file. Their split seed was 42.

| Fold | Ordinal training seed | Selected epoch |
|---:|---:|---:|
| 0 | 42 | 9 |
| 1 | 100045 | 23 |
| 2 | 200048 | 24 |
| 3 | 300051 | 3 |
| 4 | 400054 | 3 |

## Run on the pseudo dataset

Run these commands from `checkpoints/` using Python 3.12. An existing Step 2 environment can run the CPU example directly.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r ../train/teacher/requirements.txt
python example.py --seed 42
```

The CPU command loads the final ordinal model, EBM, soft MLP and reactive MLP. To also load Qwen, use a CUDA-enabled Linux environment with the Step 3 Qwen dependencies. If the base model is already cached at the pinned revision, skip the download command. Model loading runs offline after that.

```bash
python -m pip install -r ../train/distillation/requirements-qwen.txt
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
python example.py --seed 42 --qwen
```

Use `--seed 123` or `--seed 2026` for the other teacher–student pairs. The example verifies input and model hashes, reads the first synthetic validation case, applies the historical training-fitted encoder and ordinal calibration, and builds the 87-feature planner state. At the first decision only demographics are visible and the budget is $500. The policy output assigns zero probability to unavailable or unaffordable actions.

## Verified example

On 4090pc the final model assigned synthetic case `SYN-0005` the probabilities **0.058758 CN, 0.589685 MCI and 0.351557 AD**. These are the model's outputs on a fictional patient. At seed 42 all four policies chose `REQUEST_COGNITIVE`.

| Policy | Probability of the chosen action |
|---|---:|
| EBM teacher | 0.519514 |
| Soft MLP | 0.507236 |
| Reactive MLP | 0.999155 |
| Qwen soft KL | 0.507462 |

The [full seed-42 output](example_result.json) contains all ten action probabilities. On this case, the final and five fold ordinal posteriors matched their private originals exactly. The 87-dimensional planner input also matched. For every teacher and student seed, all ten probabilities agreed with the original checkpoint to an absolute tolerance of `1e-6`. Qwen ran on the RTX 4090. The CPU example was also checked from the local delivery directory.

To evaluate these weights on the complete original test cohort, use the [locked-test evaluation](../eval/README.md).

## File formats and provenance

- **Ordinal.** Each directory holds a tensor-only `weights.pt`, an ordered `preprocessing.json` and a `calibration.json`. Use the encoder and calibration from the same directory. The encoder stores fitted numeric statistics and categorical levels in training-column order. The calibration stores the positive temperature and two fitted thresholds. The original final ordinal checkpoint carried 1,552 training patient IDs, and the fold checkpoints contained membership lists. The export keeps model parameters and drops those lists.
- **EBM and MLPs.** Each `seed_*/weights.pt` is a plain PyTorch state dictionary, loaded with `weights_only=True`. The manifest provides the input dimension, hidden widths and other constructor parameters, as well as the EBM policy temperature. The trained tensor values were compared to the originals during export.
- **Qwen.** Each seed contains the original LoRA adapter tensors, its ten-action head and the settings needed to load them. All three use the byte-identical training tokenizer stored once in [the shared tokenizer directory](weights/qwen/tokenizer/). The adapter configuration refers to the pinned base model in place of the host's cache path. Obtain the 9B backbone separately with the command above.
- **Source hashes.** For ordinal, EBM and MLP entries, `source_sha256` identifies the original training checkpoint. For Qwen it identifies the original policy metadata and `source_payload_sha256` identifies the training payload. `file_sha256` covers the portable files in this package. The original training artifacts remain in private storage. Clinical tables, patient IDs, fold assignments, belief caches, prompts from research patients and training trajectories are excluded.

To repeat the export on the trusted training host, use the project environment, the original run tree and a fresh private output directory. The [export script](export_checkpoints.py) validates the original run records and writes the weights, configuration snapshots and manifest. The [inference script](example.py) uses the model classes from Steps 2 and 3.

```bash
python export_checkpoints.py --project /path/to/adni_soft_ebm_planner --output /private/new-checkpoint-export
```
