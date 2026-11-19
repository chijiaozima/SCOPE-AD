"""Reproduce the 4090pc locked-test rollouts from the released weights and a private Step 1 build."""

import argparse
import csv
from dataclasses import replace
import gc
from itertools import product
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "checkpoints"))
import example as checkpoints
from src.data.masks import expand_patient_partial_states
from src.data.preprocessing import ModalityPreprocessor, _ColumnSpec, load_and_parse
from src.evaluation.diagnosis import (
    classwise_sensitivity_specificity,
    extreme_error_rate,
    macro_f1,
    ordinal_mae,
)
from src.ordinal.cache import (
    build_canonical_modality_vectors,
    canonical_implementation_sha256,
    canonical_input_schema_sha256,
    canonical_ordinal_coordinates,
    compose_canonical_h,
)
from src.ordinal.crossfit import predict_ordinal_beliefs
from src.ordinal.model import OrdinalBeliefModel
from src.planner.costs import RAW_USD_COSTS
from src.planner.ebm import EnergyNetwork
from src.states.action_space import ACTIONS, MODALITIES, valid_action_mask
from src.states.state_builder import PlannerState
from src.states.subset_lattice import encode_mask
from src.students.mlp_policy import MLPPolicy
from src.students.prompts import build_structured_prompt
from src.utils.seed import set_seed
from scripts.generate_planner_trajectories import _observed_information

BRANCH = {
    "soft_ebm": "ebm",
    "mlp_distilled": "mlp",
    "reactive_mlp": "reactive_mlp",
    "llm_soft": "qwen",
}
SEEDS = (42, 123, 2026)
BUDGETS = (500, 1000, 2000, 4000, 8520)
KEY = ("method", "seed", "budget_usd")
CHECKPOINTS = checkpoints.ROOT


def test_states(data, manifest, device):
    reference = checkpoints.read(PROJECT / "data_process/reference.json")
    for name, record in reference["final_files"].items():
        checkpoints.verify(data / name, record["sha256"])
    cfg = yaml.safe_load((CHECKPOINTS / "configs/data.yaml").read_text())
    cfg.update(input_path=str(data), locked_split_manifest=str(data / "split_ids_seed42.csv"))
    episodes = load_and_parse(cfg)
    test = episodes.indices("test")
    if len(test) != reference["counts"]["test"]:
        raise ValueError("Unexpected number of test episodes")
    ordinal = next(e for e in manifest["checkpoints"] if e["branch"] == "ordinal")
    folder = CHECKPOINTS / ordinal["directory"]
    specs = checkpoints.read(folder / "preprocessing.json")
    prep = ModalityPreprocessor()
    prep.input_dims = ordinal["input_dims"]
    prep.specs = {m: {k: _ColumnSpec(**v) for k, v in cols.items()} for m, cols in specs.items()}
    prepared = prep.transform(episodes)
    frames = {
        m: episodes.modality_frames[m].reindex(columns=list(cols)) for m, cols in specs.items()
    }
    if (
        canonical_implementation_sha256() != manifest["canonical_encoder_sha256"]
        or canonical_input_schema_sha256(frames) != manifest["canonical_input_schema_sha256"]
    ):
        raise ValueError("Canonical planner encoder or feature schema differs from training")
    rows, masks, _ = expand_patient_partial_states(
        test, prepared.available_mask, max_states_per_patient=128
    )
    vectors = compose_canonical_h(build_canonical_modality_vectors(frames), rows, masks)
    model = OrdinalBeliefModel(prep.input_dims, **ordinal["model_config"])
    model.load_state_dict(checkpoints.state_dict(folder / "weights.pt"))
    cal = checkpoints.read(folder / "calibration.json")
    # Match the original full-test enumeration and 256-state CUDA inference batches.
    pred = predict_ordinal_beliefs(
        model,
        prepared,
        rows,
        masks,
        batch_size=256,
        device=device,
        temperature=cal["temperature"],
        calibrated_thresholds=[cal["calibrated_kappa1"], cal["calibrated_kappa2"]],
    )
    posterior = pred["probabilities"]
    severity, distances = canonical_ordinal_coordinates(posterior)
    states = {}
    for j, (i, mask) in enumerate(zip(rows, masks, strict=True)):
        states[int(i), encode_mask(mask)] = PlannerState(
            patient_id=str(prepared.patient_ids[i]),
            h=tuple(vectors[j].tolist()),
            severity=float(severity[j]),
            posterior=tuple(posterior[j].tolist()),
            entropy=float(pred["entropy"][j]),
            boundary_distances=tuple(distances[j].tolist()),
            observed_mask=tuple(mask.tolist()),
            available_mask=tuple(prepared.available_mask[i].tolist()),
        )
    return prepared, test, states


def load_policy(entry, device):
    folder = CHECKPOINTS / entry["directory"]
    branch = entry["branch"]
    if branch == "qwen":
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from src.students.llm_action_policy import load_saved_policy

        snapshot = snapshot_download(
            entry["base_model"], revision=entry["revision"], local_files_only=True
        )
        model, _ = load_saved_policy(snapshot, folder, device_map={"": 0})
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            CHECKPOINTS / entry["tokenizer"], local_files_only=True
        )

        def qwen(state, valid, features, mask, info):
            prompt = build_structured_prompt(
                dict(
                    belief_mode="ordinal",
                    observed_information=info,
                    h=list(state.h),
                    severity=state.severity,
                    posterior=list(state.posterior),
                    entropy=state.entropy,
                    distance_to_cn_mci_boundary=state.boundary_distances[0],
                    distance_to_mci_ad_boundary=state.boundary_distances[1],
                    observed_mask=list(state.observed_mask),
                    available_mask=list(state.available_mask),
                    cumulative_cost_usd=state.cumulative_cost,
                    remaining_budget_usd=state.remaining_budget,
                    candidate_costs_usd=dict(RAW_USD_COSTS),
                    action_history=list(state.action_history),
                    valid_action_mask=valid.tolist(),
                    step_index=len(state.action_history),
                    max_acquisitions=7,
                ),
                max_chars=entry["prompt_max_chars"],
            )
            tokens = tokenizer(prompt, return_tensors="pt", truncation=False)
            if tokens.input_ids.shape[-1] > entry["max_length"]:
                raise ValueError("A decision prompt exceeds the original model context")
            return model(tokens.input_ids.to(device), tokens.attention_mask.to(device), mask)[
                "probabilities"
            ]

        return qwen

    cfg = entry["model_config"]
    if branch == "ebm":
        model = EnergyNetwork(cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"])
    else:
        model = MLPPolicy.from_config(cfg)
    model.load_state_dict(checkpoints.state_dict(folder / "weights.pt"))
    model.to(device).eval()
    if branch == "ebm":
        return lambda state, valid, x, mask, info: model.policy(
            x, mask, temperature=entry["energy_temperature"]
        )
    return lambda state, valid, x, mask, info: model.policy(x, mask).probabilities


def probabilities(values, valid):
    p = values.detach().float().cpu().numpy().reshape(-1).astype(float)
    if (
        p.shape != valid.shape
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p[~valid] != 0)
        or not np.isclose(p.sum(), 1.0, atol=1e-5)
    ):
        raise ValueError("Invalid masked action distribution")
    return p


@torch.inference_mode()
def evaluate(method, seed, budget, prepared, test, states, teacher, policy, device):
    predictions, costs, requests, agrees, kls = [], [], [], [], []
    for i in test:
        code, cost, history = 0, 0.0, []
        for _ in range(len(MODALITIES) + 1):
            state = replace(
                states[i, code],
                cumulative_cost=cost,
                remaining_budget=budget - cost,
                action_history=tuple(history),
            )
            valid = valid_action_mask(
                state.observed_mask,
                state.available_mask,
                cumulative_cost=cost,
                budget=budget,
                raw_costs=RAW_USD_COSTS,
                max_acquisitions=7,
            )
            features = torch.from_numpy(state.feature_vector()).unsqueeze(0).to(device)
            mask = torch.from_numpy(valid).unsqueeze(0).to(device)
            teacher_p = probabilities(teacher(state, valid, features, mask, None), valid)
            info = (
                _observed_information(prepared, i, state.observed_mask)
                if method == "llm_soft"
                else None
            )
            p = (
                teacher_p
                if method == "soft_ebm"
                else probabilities(policy(state, valid, features, mask, info), valid)
            )
            action = int(p.argmax())
            agrees.append(action == int(teacher_p.argmax()))
            kls.append(
                float(np.sum(teacher_p * (np.log(teacher_p.clip(1e-12)) - np.log(p.clip(1e-12)))))
            )
            if action >= len(MODALITIES):
                predictions.append(action - len(MODALITIES))
                costs.append(cost)
                requests.append(len(history))
                break
            history.append(ACTIONS[action])
            code |= 1 << action
            cost += RAW_USD_COSTS[MODALITIES[action]]
        else:
            raise ValueError("An episode exhausted its actions without a diagnosis")
    true = prepared.labels[test]
    pred = np.asarray(predictions)
    recall, _ = classwise_sensitivity_specificity(true, pred)
    return dict(
        method=method,
        seed=seed,
        budget_usd=budget,
        n_test=len(test),
        cn_recall=recall["CN"],
        mci_recall=recall["MCI"],
        ad_recall=recall["AD"],
        macro_f1=macro_f1(true, pred),
        accuracy=float((pred == true).mean()),
        avg_cost=np.mean(costs),
        avg_requests=np.mean(requests),
        mdd=ordinal_mae(true, pred),
        severe_error_rate=extreme_error_rate(true, pred),
        invalid_request_rate=0.0,
        teacher_action_agreement=np.mean(agrees),
        teacher_kl=np.mean(kls),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, required=True, help="Private directory produced by prepare_adni.py"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New aggregate CSV outside the project"
    )
    parser.add_argument("--methods", choices=BRANCH, nargs="+", default=list(BRANCH))
    parser.add_argument("--seeds", choices=SEEDS, type=int, nargs="+", default=SEEDS)
    parser.add_argument("--budgets", choices=BUDGETS, type=int, nargs="+", default=BUDGETS)
    args = parser.parse_args()
    data, output = args.data.expanduser().resolve(), args.output.expanduser().resolve()
    if (
        data.is_relative_to(PROJECT)
        or output.is_relative_to(PROJECT)
        or output.is_relative_to(data)
    ):
        parser.error(
            "Keep research input and output outside this project and write results outside the dataset"
        )
    if output.exists() or args.output.expanduser().is_symlink():
        parser.error("Choose a new output CSV")
    if any(len(x) != len(set(x)) for x in [args.methods, args.seeds, args.budgets]):
        parser.error("Methods, seeds and budgets must be unique")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if "llm_soft" in args.methods and device.type != "cuda":
        parser.error("The Qwen branch requires CUDA and the pinned base model")
    with (HERE / "reference_test.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        reference = {(r["method"], int(r["seed"]), int(float(r["budget_usd"]))): r for r in reader}
    if set(reference) != set(product(BRANCH, SEEDS, BUDGETS)):
        raise ValueError("Incomplete historical reference grid")

    manifest = checkpoints.read(CHECKPOINTS / "manifest.json")
    for name, expected in manifest["file_sha256"].items():
        checkpoints.verify(CHECKPOINTS / name, expected)
    if manifest["actions"] != list(ACTIONS):
        raise ValueError("Unexpected checkpoint action order")
    entries = {(e["branch"], e["seed"]): e for e in manifest["checkpoints"]}
    set_seed(42)
    prepared, test, states = test_states(data, manifest, device)
    # Deterministic ordinal cache construction and the separate historical rollout process
    # have different Torch settings. Restore the rollout process default for Qwen kernels.
    torch.use_deterministic_algorithms(False)
    print(
        f"Verified locked split and weights. Evaluating {len(test)} test cases on {device}.",
        flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    mismatches = []
    with output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for seed in args.seeds:
            teacher = load_policy(entries["ebm", seed], device)
            for method in args.methods:
                policy = (
                    teacher
                    if method == "soft_ebm"
                    else load_policy(entries[BRANCH[method], seed], device)
                )
                for budget in args.budgets:
                    row = evaluate(
                        method, seed, budget, prepared, test, states, teacher, policy, device
                    )
                    expected = reference[method, seed, budget]
                    failed = [
                        c
                        for c in fields
                        if c not in KEY
                        and not np.isclose(
                            row[c],
                            float(expected[c]),
                            rtol=0,
                            atol=1e-6 if c == "teacher_kl" else 1e-12,
                        )
                    ]
                    writer.writerow(row)
                    f.flush()
                    print(
                        f"{method} seed={seed} budget={budget}: "
                        + ("differs: " + ", ".join(failed) if failed else "matched"),
                        flush=True,
                    )
                    mismatches.extend((method, seed, budget, c) for c in failed)
                del policy
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    if mismatches:
        raise SystemExit(
            f"{len(mismatches)} metric comparisons differ; inspect the aggregate output"
        )
    print(
        f"All {len(args.methods)*len(args.seeds)*len(args.budgets)} requested rows match. Output: {output}"
    )


if __name__ == "__main__":
    main()
