"""Canonical deduplication and training-partition-only descriptor scaling."""
import copy
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from .constants import ENDPOINT_FILES

DESCRIPTOR_FUNCTIONS = (
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("MolMR", Descriptors.MolMR),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("RingCount", Descriptors.RingCount),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
    ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ("NumSaturatedRings", Descriptors.NumSaturatedRings),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("NHOHCount", Descriptors.NHOHCount),
    ("NOCount", Descriptors.NOCount),
    ("NumValenceElectrons", Descriptors.NumValenceElectrons),
    ("LabuteASA", Descriptors.LabuteASA),
    ("BalabanJ", Descriptors.BalabanJ),
    ("BertzCT", Descriptors.BertzCT),
    ("MaxPartialCharge", Descriptors.MaxPartialCharge),
    ("MinPartialCharge", Descriptors.MinPartialCharge),
    ("MaxAbsPartialCharge", Descriptors.MaxAbsPartialCharge),
    ("MinAbsPartialCharge", Descriptors.MinAbsPartialCharge),
)


def canonical_smiles(smiles):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def murcko_group(canonical):
    molecule = Chem.MolFromSmiles(canonical)
    if molecule is None:
        raise ValueError("Invalid canonical SMILES")
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=False)
    # Acyclic molecules must remain separate scaffold groups.
    return scaffold_smiles if scaffold_smiles else "ACYCLIC:" + canonical


def make_endpoint_dataset(data_root, endpoint, graph_map):
    csv_path = Path(data_root) / ENDPOINT_FILES[endpoint]
    raw = pd.read_csv(csv_path)
    if not {"SMILES", "Label"}.issubset(raw.columns):
        raise ValueError("{} must have SMILES and Label columns".format(csv_path))
    audit, candidates = Counter({"input_rows": len(raw)}), []
    for row_index, row in raw.iterrows():
        try:
            numeric_label = float(row["Label"])
        except (ValueError, TypeError, OverflowError):
            audit["invalid_label"] += 1
            continue
        if not np.isfinite(numeric_label) or numeric_label not in (0., 1.):
            audit["nonbinary_label"] += 1
            continue
        label = int(numeric_label)
        canonical = canonical_smiles(row["SMILES"])
        if canonical is None:
            audit["invalid_smiles"] += 1
            continue
        if canonical not in graph_map:
            audit["missing_simg_graph"] += 1
            continue
        candidates.append({
            "canonical_smiles": canonical,
            "label": label,
            "scaffold": murcko_group(canonical),
            "row_index": int(row_index),
        })
    labels_by_canonical = {}
    for candidate in candidates:
        labels_by_canonical.setdefault(candidate["canonical_smiles"], set()).add(candidate["label"])
    conflict_set = {key for key, values in labels_by_canonical.items() if len(values) > 1}
    records, seen = [], set()
    for candidate in candidates:
        canonical = candidate["canonical_smiles"]
        if canonical in conflict_set:
            audit["conflicting_canonical_rows_excluded"] += 1
            continue
        if canonical in seen:
            audit["duplicate_smiles_same_label"] += 1
            continue
        seen.add(canonical)
        records.append(candidate)
    if len({record["label"] for record in records}) != 2:
        raise ValueError("{} has fewer than two classes after SIMG matching".format(endpoint))
    audit["usable_records"] = len(records)
    audit["usable_positive"] = int(sum(record["label"] for record in records))
    audit["usable_negative"] = int(len(records) - audit["usable_positive"])
    audit["unique_scaffolds"] = len({record["scaffold"] for record in records})
    audit_payload = dict(audit)
    audit_payload["conflicting_canonical_count"] = len(conflict_set)
    audit_payload["conflicting_canonical_smiles_excluded"] = sorted(conflict_set)
    return records, audit_payload


def build_graph_list(records, graph_map):
    graph_list = []
    for record in records:
        graph = copy.deepcopy(graph_map[record["canonical_smiles"]])
        graph.y = torch.tensor([record["label"]], dtype=torch.long)
        graph_list.append(graph)
    return graph_list


def build_descriptor_matrix(records):
    matrix = []
    for record in records:
        molecule = Chem.MolFromSmiles(record["canonical_smiles"])
        values = [float(function(molecule)) for _, function in DESCRIPTOR_FUNCTIONS]
        matrix.append(values)
    # A few RDKit charge descriptors can be non-finite for chemically valid
    # molecules. Keep the samples and defer imputation to the training-fold
    # scaler, rather than excluding samples or inspecting test-fold statistics.
    matrix = np.asarray(matrix, dtype=np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def fit_descriptor_scaler(descriptor_matrix, train_indices):
    values = descriptor_matrix[np.asarray(train_indices, dtype=int)]
    mean = np.nanmean(values, axis=0)
    if not np.isfinite(mean).all():
        invalid = [DESCRIPTOR_FUNCTIONS[i][0] for i, value in enumerate(mean) if not np.isfinite(value)]
        raise ValueError("A training fold has no finite values for: {}".format(", ".join(invalid)))
    imputed_values = np.where(np.isfinite(values), values, mean)
    scale = imputed_values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {
        "names": [name for name, _ in DESCRIPTOR_FUNCTIONS],
        "mean": mean,
        "scale": scale,
        "imputation_scope": "training_partition_finite_mean_only",
    }


def transform_descriptors(descriptor_matrix, scaler):
    imputed = np.where(np.isfinite(descriptor_matrix), descriptor_matrix, scaler["mean"])
    transformed = (imputed - scaler["mean"]) / scaler["scale"]
    if not np.isfinite(transformed).all():
        raise ValueError("Descriptor scaling created non-finite values")
    return transformed.astype(np.float32)


def prepare_graph_subset(graph_list, indices, scaled_descriptors=None):
    subset = []
    for index in indices:
        graph = copy.deepcopy(graph_list[int(index)])
        if scaled_descriptors is not None:
            graph.descriptor = torch.tensor(scaled_descriptors[int(index)], dtype=torch.float32).view(1, -1)
        subset.append(graph)
    return subset
