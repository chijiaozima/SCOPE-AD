# Locked ADNI test evaluation

Run this evaluation to reproduce the 4090pc policy results using the released checkpoints and the original 333 test participants. All inputs needed from ADNI are produced by [Step 1](../data_process/README.md). Model loading uses the [checkpoint manifest](../checkpoints/manifest.json). No training, private belief cache or teacher trajectories are required.

## Full run

From the project root, activate the Python 3.12 model environment, install the GPU dependencies and cache the pinned Qwen backbone. Use a new output filename in private storage. The default evaluates all 60 combinations.

```bash
source ../cdm-env/bin/activate
python -m pip install -r requirements-qwen.txt
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
HF_HUB_OFFLINE=1 python eval/evaluate.py \
  --data /private/study/derived/4090pc-seed42 \
  --output /private/study/results/locked-test.csv
```

The original device was an RTX 4090. Use the pinned environment and the same device for numerical comparisons, particularly the 4-bit Qwen branch. To check a smaller slice, specify any subset of the study methods, seeds and budgets. This example also runs in the CPU environment.

```bash
python eval/evaluate.py \
  --data /private/study/derived/4090pc-seed42 \
  --methods soft_ebm mlp_distilled reactive_mlp --seeds 42 --budgets 1000 \
  --output /private/study/results/seed42-1000.csv
```

| Method argument | Research checkpoint |
|---|---|
| `soft_ebm` | Soft EBM teacher |
| `mlp_distilled` | MLP student trained on soft policy targets |
| `reactive_mlp` | MLP student trained on the teacher's hard action |
| `llm_soft` | Qwen3.5-9B student, soft policy targets, seed-specific adapter and action head |

## What the command checks

1. Verify the SHA-256 of all four private split files against the frozen reference and verify the released model/configuration files. The final ordinal model always uses seed 42. The EBM and each student use the requested policy seed.
2. Load the training-fitted preprocessor and calibrated ordinal model. Rebuild the partial-observation states of the 333 held-out participants. At the first decision only demographics are observed. For each policy step, mask unaffordable, unavailable and already requested tests, then choose the highest-probability valid action. Charge each acquired test at its recorded dollar cost. End the episode when the policy chooses a diagnosis. Labels enter only the final scoring calculation. [Dollar costs](../checkpoints/configs/costs.yaml)
3. Write one aggregate row per method, seed and budget. Compare each row directly with [the 60 historical rows](reference_test.csv). A complete default run prints `All 60 requested rows match` and exits with status zero. A mismatch names the affected columns and returns a nonzero exit status. An interrupted run can leave a partial CSV. Use the final success message to confirm completeness.

The study used seeds **42, 123 and 2026** and per-person budgets of **$500, $1,000, $2,000, $4,000 and $8,520**. The same test participants appear in each setting. The CSV contains sample count, recall for each diagnosis, macro-F1, accuracy, mean spend, mean requests, mean absolute ordinal diagnosis error (`mdd`), severe-error rate (CN ↔ AD), invalid-request rate, teacher action agreement and teacher-to-policy KL. Agreement and KL average over the evaluated policy's own visited decision states, using its seed-matched EBM as teacher. The absolute comparison tolerance is `1e-12` for all columns except KL, which uses `1e-6`. The evaluation writes aggregate metrics only.

## Historical comparison

At the primary **$1,000** budget the unweighted means across the three policy seeds are:

| Model | Macro-F1 | Accuracy | Mean spend ($) | Mean requests |
|---|---:|---:|---:|---:|
| EBM teacher | 0.774920 | 0.775776 | 51.625626 | 1.007007 |
| Soft MLP | 0.775746 | 0.776777 | 51.393393 | 1.006006 |
| Hard-action MLP | 0.777764 | 0.776777 | 51.393393 | 1.006006 |
| Qwen soft-policy student | 0.777148 | 0.776777 | 50.464464 | 1.002002 |

The per-seed CSV was assembled from the archived August 2026 test episode and decision records **before** running the public evaluator. For EBM, soft MLP and Qwen, the common columns match all 45 policy rows of the original `results/pareto_budget.csv` and all nine rows of `results/distillation_comparison.csv` at $1,000. The hard-action MLP's 15 rows come from its archived `main_table_nonllm` rollouts. That branch was evaluated on 4090pc and did not appear in those two summary tables.

### Original training-host commands

The original research tree used the following entry points. Its non-LLM configuration also ran comparison baselines on validation and test. The temporary Qwen configuration ran the trained soft-policy student on test. Both audits recorded the same three seeds and five budgets. The Qwen audit names a temporary file at `/tmp/qwen_soft_test_rollout.yaml`. This release includes the [original non-LLM YAML](4090pc_configs/rollout_main_table_nonllm.yaml) and a [Qwen YAML reconstructed from the completed run](4090pc_configs/rollout_qwen_soft_test.yaml). For archival replay, place both files under `configs/` in a **private working copy** of the original tree alongside its `costs.yaml` and research artifacts. Its scripts write patient-level artifacts. The public command above takes the verified Step 1 files and produces the aggregate comparison directly.

```bash
# From the original private adni_soft_ebm_planner tree
python scripts/run_rollouts.py --config configs/rollout_main_table_nonllm.yaml
python scripts/run_rollouts.py --config configs/rollout_qwen_soft_test.yaml
python scripts/build_experiment_summary.py
```

| Private reference artifact | SHA-256 as checked on 4090pc |
|---|---|
| main_table_nonllm/rollout_episodes.parquet | `0d3114ea8633a4ac25689f89cddaeb1bdd5a53460237de45ea50d0135c93e2ad` |
| main_table_nonllm/rollout_steps.parquet | `f445a79c474a06f11f209aa0e4d37431d968ae1e1f5f45bad1a0adf9b253d36a` |
| qwen_soft_test/rollout_episodes.parquet | `c3419e5829c7744ccf154f61b96077f4060c45a39380f447ee819bfdb7c90881` |
| qwen_soft_test/rollout_steps.parquet | `57e1bc8c4caf49183ad4948e7631e0ea93022c2b2b2e4fd4cd8ad13be80abc61` |
| results/pareto_budget.csv | `0d5ef9a9c556dc27641fef6b6a9293702bb451eacce770d71ee6f7499f6fd564` |
| results/distillation_comparison.csv | `e45f558f4fa3b63b37f24153dc50e070f22e222048e6d4a426fc75d48dcd656b` |

## Release verification on 19 November 2026

The public evaluator ran on the original RTX 4090 using Python 3.12.3, PyTorch 2.13.0+cu130 and the pinned Qwen dependencies. All **60/60** reference rows passed on the locked 333-person test set, covering **19,980 episodes**. The largest absolute difference across the sample count, diagnosis, acquisition and teacher-agreement metrics was `0`. The largest absolute teacher-KL difference was `8.88e-16`. A separate clean CPU environment reproduced all 45 EBM/MLP rows, with maximum absolute KL difference `1.33e-07`.
