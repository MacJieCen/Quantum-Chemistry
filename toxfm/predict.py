"""Strict loading and stage-wise scoring of versioned checkpoint bundles."""
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from .constants import GRAPH_SCHEMA, SUBTYPES
from .features import classifier_graph
from .io import sha256
from .model import ARCHITECTURE, FormalSIMGHouseholderParseCaps
from .preprocessing import build_descriptor_matrix, transform_descriptors, DESCRIPTOR_FUNCTIONS


class TwoStagePredictor:
    def __init__(self, model_dir, endpoints=SUBTYPES, device="cpu"):
        self.root = Path(model_dir).resolve()
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format_version") != 1 or manifest.get("architecture") != ARCHITECTURE:
            raise ValueError("Unsupported checkpoint bundle architecture or format")
        if manifest.get("graph_schema") != GRAPH_SCHEMA:
            raise ValueError("Bundle SIMG feature schema does not match the predictor")
        self.manifest = manifest
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; use --device cpu")
        self.device = torch.device(device)
        self.models = {}
        for endpoint in ("FirstStage",) + tuple(endpoints):
            entries = manifest["endpoints"].get(endpoint, [])
            if not entries:
                raise ValueError("No checkpoint for {}".format(endpoint))
            self.models[endpoint] = []
            for entry in entries:
                path = (self.root / entry["file"]).resolve()
                if self.root not in path.parents:
                    raise ValueError("Checkpoint path must remain inside --model-dir")
                if sha256(path) != entry["sha256"]:
                    raise ValueError("Checkpoint SHA-256 mismatch: {}".format(entry["file"]))
                checkpoint = torch.load(path, map_location="cpu")
                if checkpoint.get("endpoint") != endpoint or checkpoint.get("graph_schema") != GRAPH_SCHEMA:
                    raise ValueError("Checkpoint endpoint/schema mismatch")
                if checkpoint.get("architecture") != ARCHITECTURE:
                    raise ValueError("Checkpoint architecture mismatch")
                threshold = float(checkpoint["threshold"])
                if not 0 <= threshold <= 1 or threshold != float(entry["threshold"]):
                    raise ValueError("Invalid or inconsistent validation threshold")
                model = FormalSIMGHouseholderParseCaps(**checkpoint["model_config"])
                model.load_state_dict(checkpoint["model_state"], strict=True)
                model.to(self.device).eval()
                scaler = checkpoint.get("descriptor_scaler")
                if model.descriptor_dim:
                    if scaler is None or scaler["names"] != [n for n, _ in DESCRIPTOR_FUNCTIONS]:
                        raise ValueError("Missing or incompatible descriptor scaler")
                    scaler = dict(scaler, mean=np.asarray(scaler["mean"], dtype=np.float32),
                                  scale=np.asarray(scaler["scale"], dtype=np.float32))
                    if not np.isfinite(scaler["scale"]).all() or (scaler["scale"] <= 0).any():
                        raise ValueError("Invalid descriptor scaler")
                self.models[endpoint].append((model, threshold, entry["file"], scaler))

    @torch.no_grad()
    def _score(self, graph, endpoint, canonical_smiles):
        scores, votes, folds = [], [], []
        raw_descriptors = None
        for model, threshold, filename, scaler in self.models[endpoint]:
            prepared = classifier_graph(graph)
            if scaler is not None:
                if raw_descriptors is None:
                    raw_descriptors = build_descriptor_matrix([{"canonical_smiles": canonical_smiles}])
                prepared.descriptor = torch.tensor(transform_descriptors(raw_descriptors, scaler))
            batch = Batch.from_data_list([prepared]).to(self.device)
            score = float(model(batch)[1].item())
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError("Non-finite or out-of-range model score")
            scores.append(score)
            votes.append(score >= threshold)
            folds.append({"checkpoint": filename, "score": score,
                          "validation_threshold": threshold,
                          "positive": bool(votes[-1])})
        return {"score_mean": float(np.mean(scores)),
                "score_sd_across_models": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "positive_vote_fraction": float(np.mean(votes)),
                "positive_by_majority_vote": bool(sum(votes) > len(votes) / 2),
                "n_models": len(scores), "per_model": folds}

    def predict(self, graph, provenance):
        first = self._score(graph, "FirstStage", provenance["canonical_smiles"])
        screen_passed = first["positive_by_majority_vote"]
        second = {}
        for endpoint in self.models:
            if endpoint == "FirstStage":
                continue
            result = self._score(graph, endpoint, provenance["canonical_smiles"])
            result["screen_passed"] = screen_passed
            result["cascade_positive"] = bool(screen_passed and result["positive_by_majority_vote"])
            second[endpoint] = result
        return {"canonical_smiles": provenance["canonical_smiles"],
                "model_bundle": self.manifest["bundle_name"],
                "stage1": first, "stage2": second,
                "features": provenance}
