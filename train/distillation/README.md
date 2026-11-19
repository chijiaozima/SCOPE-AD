# Step 3 · Distill the EBM teachers into student policies

This package continues a completed [Step 2 teacher run](../teacher/README.md). It generates action trajectories from the three trained EBM teachers, then trains one student per teacher seed. The 4090pc study completed three student variants: a soft-policy MLP, a hard-action MLP baseline, and a Qwen3.5-9B soft-policy student.

## Minimal CPU example

From `train/distillation/`, use Python 3.12 and the same environment as Step 2. If you already ran the Step 2 example, point `--teacher-run` at its existing output directory.

```bash
python -m pip install -r requirements.txt
python run_distillation.py --teacher-run ../../../cdm-runs/teacher --output ../../../cdm-runs/students-mlp
```

If you are starting from the public files alone, create the synthetic teachers once with the following command, then run the command above. The pseudo dataset, prepared episodes, fitted preprocessor, subset states and three selected teachers must remain at their original locations. The teacher manifest records absolute paths and SHA-256 digests. Each new distillation run needs a fresh output directory.

```bash
python ../teacher/run_teacher.py --profile example --output ../../../cdm-runs/teacher
```

The CPU command generates trajectories and trains **both MLP variants** for one epoch per seed. It records the effective configurations in `configs/`, the stage logs in `logs/`, and the run products in `artifacts/` under the output directory. After training it reloads the seed-42 students and checks their action probabilities on a held-out validation state. The resulting `example_result.json` also records the selected epoch and validation loss for every seed.

## Include the Qwen student

The GPU example uses the same teacher run and settings as the CPU example, with one additional training stage. The tested environment is Linux with an RTX 4090, CUDA-enabled PyTorch and a locally cached Qwen snapshot. Install the additional dependencies in that environment. If the specified revision is already cached, skip the download command.

```bash
python -m pip install -r requirements-qwen.txt
hf download Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
python run_distillation.py --students all --teacher-run ../../../cdm-runs/teacher --output ../../../cdm-runs/students-all
```

`--students qwen` trains just the Qwen branch after trajectory generation. The trainer uses local files only and verifies the base model revision. It trains a LoRA adapter and a ten-action head. The final example check reloads the seed-42 adapter and action head and queries them on the same validation state as the MLPs.

## What is being distilled

| Stage or student | Inputs and supervision | Implementation |
|---|---|---|
| Generate trajectories | For each teacher seed, training budget and training or validation patient, start with demographics. Mask unavailable, already acquired and unaffordable requests. Normalize the negative EBM energies over valid actions at temperature 0.3. Follow the highest-probability action, reveal a requested result, and replan until a terminal diagnosis. Each visited state yields one row with its masked policy, energies and chosen action. | [generate_planner_trajectories.py](scripts/generate_planner_trajectories.py) |
| Soft-policy MLP | Maps the 87 state features to ten action logits with hidden widths 256 and 128. Fits the teacher's full masked distribution by minimizing KL divergence at distillation temperature 1.0. Hard-action and energy-margin weights are both zero. | [train_mlp_student.py](scripts/train_mlp_student.py) |
| Reactive MLP baseline | Uses the same state features, action mask, architecture and trajectories. Fits the teacher's single chosen action by minimizing negative log-likelihood. The MLP entry point trains both variants. | [MLP policy](src/students/mlp_policy.py) · [training losses](src/students/losses.py) |
| Qwen soft-policy student | Rebuilds a structured prompt from the observable state, checks it against the stored prompt, and verifies that it fits within 1,536 tokens. A ten-action head on the last non-padding token predicts the teacher distribution under the same mask. The 9B backbone uses 4-bit NF4 with double quantization, bfloat16 compute and all-linear LoRA (rank 8, alpha 16, dropout 0.05). | [train_llm_student.py](scripts/train_llm_student.py) · [Qwen action policy](src/students/llm_action_policy.py) · [prompt builder](src/students/prompts.py) |

Each student learns from the teacher with the same seed (42, 123 or 2026). Training uses only training-patient trajectories. Validation loss selects the epoch independently for each seed and branch. The generator excludes test patients before rolling out teachers. Ground-truth diagnoses, unobserved modality values and oracle values are excluded from student supervision.

## Example and historical runs

The verified example used the three existing teachers from the [published Step 2 example run](../teacher/example_result.json). It reused that run's fitted preprocessor, prepared episodes and subset-state cache. The distillation result records the SHA-256 of its teacher manifest.

| Setting | Public pseudo example | Recorded 4090pc study |
|---|---|---|
| Patients used (train / validation) | 42 / 9 | 1,552 / 333 |
| Trajectory rows (train / validation / test) | 339 / 73 / 0 | 46,939 / 10,046 / 0 |
| Budgets in USD | 500 | 500, 1,000, 2,000, 4,000, 8,520 |
| Maximum epochs (each MLP / Qwen) | 1 / 1 | 100 / 3 |
| Teacher seeds | 42, 123, 2026 | 42, 123, 2026 |

A trajectory row is one decision, so the same patient can contribute several rows. In the example the seed-42 teacher requested a cognitive assessment as its first action on synthetic validation patient `SYN-0005`. After one epoch, the reactive MLP and Qwen also chose `REQUEST_COGNITIVE` (probabilities 0.330 and 0.551). The soft MLP chose `DIAGNOSE_CN` (0.414). The example checks that training and masked inference work. Its action agreement and losses do not estimate clinical performance. [Verified synthetic run](example_result.json)

The 4090pc audits and manifests confirm the teacher trajectories and both MLP variants on 9 August 2026, and Qwen soft KL on 11 August. All three student manifests match the hashes of the same teacher manifest and trajectory file. The subsequent Qwen **hard-action** run stopped during seed 42 with exit code 139 and no completed manifest. The original entry point defaults to attempting both Qwen objectives. The packaged runner explicitly selects `soft_kl` to reproduce the completed Qwen branch. The trajectory and Qwen entry points and the prompt builder on the host were saved again on 13 August. The packaged prompt builder reproduced all 56,985 stored historical prompts exactly. The exact August training source revision is unconfirmed. [Historical run summary](research_reference.json) · [4090pc configuration](configs/distillation.yaml) · [example overrides](example.yaml)

## Use the research configuration with authorized ADNI access

Complete [Step 1](../../data_process/README.md) and [Step 2](../teacher/README.md) in private storage and keep those directories in place. From the same CUDA environment with the pinned Qwen snapshot cached, run the following command with private paths. The research profile defaults to all three completed student variants and inherits the budget and planner configuration from the supplied teacher run.

```bash
python run_distillation.py --profile research \
  --teacher-run /private/adni/teacher-run \
  --output /private/adni/student-run
```

The runner checks the Step 1 assignment digest against the Step 2 teacher manifest. The trajectory generator verifies the teacher weights, resolved planner settings, subset states, prepared data and preprocessor. Keep research outputs private: trajectories and prompts contain participant-level observations. The public package contains source, configurations and aggregate verification results. Training writes its weights only into the chosen run directory.
