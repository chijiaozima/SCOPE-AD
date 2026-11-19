"""Query the historical 4090pc checkpoints on the first synthetic validation patient."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [
    str(ROOT.parent / "train/teacher"),
    str(ROOT.parent / "train/distillation"),
]
from scripts.generate_planner_trajectories import _observed_information
from src.data.preprocessing import ModalityPreprocessor, _ColumnSpec, load_and_parse
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
from src.states.action_space import ACTIONS, valid_action_mask
from src.states.state_builder import PlannerState
from src.students.mlp_policy import MLPPolicy
from src.students.prompts import build_structured_prompt


def read(path):
    return json.loads(path.read_text())


def verify(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path.name}")


def state_dict(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def synthetic_episodes():
    data = ROOT.parent / "data_process/pseudo_dataset"
    metadata = read(data / "manifest.json")
    if metadata["synthetic"] is not True:
        raise ValueError("The example requires the Step 1 synthetic fixture.")
    for name in ("train.csv", "val.csv", "test.csv", "split_ids_seed42.csv"):
        verify(data / name, metadata["files"][name]["sha256"])
    cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    cfg.update(
        input_path=str(data),
        locked_split_manifest=str(data / "split_ids_seed42.csv"),
        locked_split_counts=metadata["counts"],
    )
    return load_and_parse(cfg)


def initial_state(ordinal, manifest):
    folder = ROOT / ordinal["directory"]
    specs = read(folder / "preprocessing.json")
    prep = ModalityPreprocessor()
    prep.input_dims = ordinal["input_dims"]
    prep.specs = {m: {k: _ColumnSpec(**v) for k, v in cols.items()} for m, cols in specs.items()}
    episodes = synthetic_episodes()
    prepared = prep.transform(episodes)
    i = int(episodes.indices("validation")[0])
    observed = np.zeros((1, 7), dtype=bool)
    model = OrdinalBeliefModel(prep.input_dims, **ordinal["model_config"])
    model.load_state_dict(state_dict(folder / "weights.pt"))
    cal = read(folder / "calibration.json")
    pred = predict_ordinal_beliefs(
        model,
        prepared,
        [i],
        observed,
        temperature=cal["temperature"],
        calibrated_thresholds=[cal["calibrated_kappa1"], cal["calibrated_kappa2"]],
    )
    posterior = pred["probabilities"]
    severity, distances = canonical_ordinal_coordinates(posterior)
    frames = {
        m: episodes.modality_frames[m].reindex(columns=list(cols)) for m, cols in specs.items()
    }
    if (
        canonical_implementation_sha256() != manifest["canonical_encoder_sha256"]
        or canonical_input_schema_sha256(frames) != manifest["canonical_input_schema_sha256"]
    ):
        raise ValueError("Unexpected planner feature encoding.")
    vectors = build_canonical_modality_vectors(frames)
    h = compose_canonical_h(vectors, np.array([i]), observed)
    state = PlannerState(
        patient_id=str(prepared.patient_ids[i]),
        h=tuple(h[0].tolist()),
        severity=float(severity[0]),
        posterior=tuple(posterior[0].tolist()),
        entropy=float(pred["entropy"][0]),
        boundary_distances=tuple(distances[0].tolist()),
        observed_mask=tuple(observed[0].tolist()),
        available_mask=tuple(prepared.available_mask[i].tolist()),
        remaining_budget=500.0,
    )
    return state, _observed_information(prepared, i, state.observed_mask)


def qwen_policy(entry, state, info, mask, valid):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    from src.students.llm_action_policy import load_saved_policy

    snapshot = snapshot_download(
        entry["base_model"], revision=entry["revision"], local_files_only=True
    )
    model, _ = load_saved_policy(snapshot, ROOT / entry["directory"], device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(ROOT / entry["tokenizer"], local_files_only=True)
    prompt = build_structured_prompt(
        {
            "belief_mode": "ordinal",
            "observed_information": info,
            "h": list(state.h),
            "severity": state.severity,
            "posterior": list(state.posterior),
            "entropy": state.entropy,
            "distance_to_cn_mci_boundary": state.boundary_distances[0],
            "distance_to_mci_ad_boundary": state.boundary_distances[1],
            "observed_mask": list(state.observed_mask),
            "available_mask": list(state.available_mask),
            "cumulative_cost_usd": state.cumulative_cost,
            "remaining_budget_usd": state.remaining_budget,
            "candidate_costs_usd": dict(RAW_USD_COSTS),
            "action_history": [],
            "valid_action_mask": valid.tolist(),
            "step_index": 0,
            "max_acquisitions": 7,
        },
        max_chars=entry["prompt_max_chars"],
    )
    batch = tokenizer([prompt], return_tensors="pt", padding=True, truncation=False)
    if batch.input_ids.shape[-1] > entry["max_length"]:
        raise ValueError("Prompt exceeds the verified context length.")
    model.eval()
    device = model.backbone.get_input_embeddings().weight.device
    with torch.inference_mode():
        return model(batch.input_ids.to(device), batch.attention_mask.to(device), mask.to(device))[
            "probabilities"
        ]


def policy_result(probs, valid):
    p = probs.detach().float().cpu().numpy().flatten()
    if (
        p.shape != (len(ACTIONS),)
        or not np.isfinite(p).all()
        or np.any(p < 0)
        or np.any(p[~valid] != 0)
        or not np.isclose(p.sum(), 1, atol=1e-5)
    ):
        raise ValueError("Action probabilities are invalid for this state.")
    return {
        "action": ACTIONS[int(p.argmax())],
        "probabilities": dict(zip(ACTIONS, p.tolist(), strict=True)),
    }


def run(seed, include_qwen):
    manifest = read(ROOT / "manifest.json")
    selected = ["ebm", "mlp", "reactive_mlp"] + (["qwen"] if include_qwen else [])
    entries = {
        e["branch"]: e
        for e in manifest["checkpoints"]
        if e["branch"] == "ordinal" or (e["branch"] in selected and e["seed"] == seed)
    }
    needed = [entries[b]["directory"] + "/" for b in ["ordinal", *selected]]
    if include_qwen:
        needed.append(entries["qwen"]["tokenizer"] + "/")
    for name, expected in manifest["file_sha256"].items():
        if name.startswith(("configs/", *needed)):
            verify(ROOT / name, expected)
    if manifest["actions"] != list(ACTIONS):
        raise ValueError("Action vocabulary mismatch.")
    state, info = initial_state(entries["ordinal"], manifest)
    valid = valid_action_mask(
        state.observed_mask,
        state.available_mask,
        cumulative_cost=0,
        budget=state.remaining_budget,
        raw_costs=RAW_USD_COSTS,
        max_acquisitions=7,
    )
    mask = torch.from_numpy(valid).unsqueeze(0)
    x = torch.from_numpy(state.feature_vector(belief_mode="ordinal")).unsqueeze(0)
    predictions = {}
    for branch in selected:
        entry = entries[branch]
        if branch == "qwen":
            probs = qwen_policy(entry, state, info, mask, valid)
        else:
            cfg = entry["model_config"]
            if branch == "ebm":
                model = EnergyNetwork(cfg["input_dim"], cfg["hidden_dims"], cfg["dropout"])
            else:
                model = MLPPolicy.from_config(cfg)
            model.load_state_dict(state_dict(ROOT / entry["directory"] / "weights.pt"))
            model.eval()
            with torch.inference_mode():
                probs = (
                    model.policy(x, mask, temperature=entry["energy_temperature"])
                    if branch == "ebm"
                    else model.policy(x, mask).probabilities
                )
        predictions[branch] = policy_result(probs, valid)
    return {
        "weights": "4090pc research run",
        "dataset": "Step 1 synthetic",
        "patient": state.patient_id,
        "ordinal_seed": entries["ordinal"]["seed"],
        "policy_seed": seed,
        "budget_usd": state.remaining_budget,
        "observed_modalities": ["demographic"],
        "state_dimension": x.shape[1],
        "ordinal_posterior": dict(zip(manifest["ordinal_labels"], state.posterior, strict=True)),
        "policies": predictions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, 123, 2026), default=42)
    parser.add_argument(
        "--qwen", action="store_true", help="Also run the 4-bit Qwen branch on CUDA."
    )
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.qwen), indent=2, allow_nan=False))
