"""Request one modality for a fictional test patient. CPU only."""

import json
from pathlib import Path


from preparation.dataset import read_episodes
from preparation.schema import ACQUISITION_MODALITIES, LABELS

DATA = Path(__file__).resolve().parent / "pseudo_dataset"
# USD amounts from the 4090pc experiment configuration.
COST = dict(
    cognitive=50, apoe=283, blood=232, csf=574, structural_mri=1431, amyloid_pet=2975, tau_pet=2975
)


def valid_actions(available, observed, budget):
    tests = [
        name
        for name in ACQUISITION_MODALITIES
        if name in available and name not in observed and COST[name] <= budget
    ]
    return tests + [f"diagnose {label}" for label in LABELS]


def main():
    metadata = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if metadata.get("synthetic") is not True:
        raise ValueError("The example expects the bundled synthetic dataset")
    episodes = read_episodes(DATA)
    if not all(str(item).startswith("SYN-") for item in episodes.patient_ids):
        raise ValueError("The example requires fictional identifiers")

    idx = episodes.indices("test")[0]
    available = {
        name for name, ok in zip(ACQUISITION_MODALITIES, episodes.available_mask[idx]) if ok
    }
    observed = set()
    budget = 500
    print(f"Synthetic patient {episodes.patient_ids[idx]}. Budget: ${budget}.")
    demographic = episodes.modality_frames["demographic"].iloc[idx].drop(labels="base_present")
    print("Visible demographics:", demographic.dropna().to_dict())
    actions = valid_actions(available, observed, budget)
    print("Valid actions:", ", ".join(actions))
    request = "cognitive"  # A scripted demonstration, independent of diagnosis.
    if request not in actions:
        raise ValueError("The example patient has no feasible cognitive assessment")
    observed.add(request)
    budget -= COST[request]
    result = episodes.modality_frames[request].iloc[idx].dropna()
    print(f"Requested {request} (${COST[request]}). Result:", result.to_dict())
    print(f"Remaining budget: ${budget}.")
    print("Valid actions now:", ", ".join(valid_actions(available, observed, budget)))


if __name__ == "__main__":
    main()
