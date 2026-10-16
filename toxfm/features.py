"""Generate genuine SIMG* graphs or read an explicitly supplied trusted cache.

The upstream LP and NBO checkpoint loaders remain strict. No RDKit-only
feature fallback, silent zero padding, salt stripping or neutralization occurs.
RDKit supplies a reproducible 3D conformer; Open Babel infers connectivity as
in the server's original XYZ -> SIMG pipeline. All choices enter provenance.
"""
import contextlib
import copy
import io
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

from .constants import GRAPH_SCHEMA
from .io import sha256


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Invalid or empty SMILES: {!r}".format(smiles))
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def validate_graph(graph):
    if not isinstance(graph, Data):
        raise ValueError("Expected a torch_geometric.data.Data SIMG graph")
    if graph.x.ndim != 2 or graph.x.shape[1] != 39 or not len(graph.x):
        raise ValueError("SIMG graph requires non-empty x with 39 columns")
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError("SIMG edge_index must have shape [2, E]")
    if graph.edge_attr.shape != (graph.edge_index.shape[1], 27):
        raise ValueError("SIMG graph requires edge_attr with 27 columns")
    if graph.edge_index.dtype != torch.long:
        raise ValueError("SIMG edge indices must be torch.long")
    if graph.edge_index.numel() and (int(graph.edge_index.min()) < 0 or int(graph.edge_index.max()) >= len(graph.x)):
        raise ValueError("SIMG edge references a nonexistent node")
    if not torch.isfinite(graph.x).all() or not torch.isfinite(graph.edge_attr).all():
        raise ValueError("SIMG features contain non-finite values")
    return graph


def classifier_graph(graph, label=None):
    """Keep exactly the features consumed by the classifier; never treat y as x."""
    validate_graph(graph)
    out = Data(x=graph.x.detach().cpu().float().clone(),
               edge_index=graph.edge_index.detach().cpu().long().clone(),
               edge_attr=graph.edge_attr.detach().cpu().float().clone())
    # Preserve node identity, orbital interaction edges and coordinates used by
    # the v2 encoder. PyG offsets interaction_edge_index when batching graphs.
    for name in ("is_atom", "is_lp", "is_bond", "xyz_data", "interaction_edge_index"):
        value = getattr(graph, name, None)
        if value is None:
            raise ValueError("SIMG* graph is missing required field: " + name)
        out[name] = value.detach().cpu().clone()
    membership = torch.stack([out.is_atom, out.is_lp, out.is_bond], dim=-1)
    if membership.shape != (len(out.x), 3) or not bool((membership.sum(-1) == 1).all()):
        raise ValueError("SIMG* node-type membership is invalid")
    if out.xyz_data.shape != (len(out.x), 3) or not torch.isfinite(out.xyz_data).all():
        raise ValueError("SIMG* node coordinates are invalid")
    if out.interaction_edge_index.numel() == 0:
        out.interaction_edge_index = torch.empty((2, 0), dtype=torch.long)
    edges = out.interaction_edge_index
    if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("Invalid interaction-edge indices")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= len(out.x)):
        raise ValueError("Interaction edge references a nonexistent node")
    if label is not None:
        out.y = torch.tensor([int(label)], dtype=torch.long)
    return out


def read_graph_cache(path):
    """Only deserialize trusted project pickle/torch files, never public uploads."""
    path = Path(path)
    if path.suffix in (".pt", ".pth"):
        raw = torch.load(path, map_location="cpu")
    else:
        with path.open("rb") as handle:
            raw = pickle.load(handle)
    if isinstance(raw, dict) and "graphs" in raw:
        raw = raw["graphs"]
    if isinstance(raw, dict):
        pairs = raw.items()
    else:
        pairs = ((item[0], item[2][0]) for item in raw)
    result = {}
    for smi, graph in pairs:
        try:
            key = canonical_smiles(smi)
        except ValueError:
            continue
        validate_graph(graph)
        # Original trusted cache may contain repeated conformers. Keep the first,
        # matching the established formal-training loader; do not overwrite it.
        if key not in result:
            result[key] = graph
    return result


class SIMGFeaturizer:
    def __init__(self, simg_root=None, graph_cache=None, seed=42):
        self.simg_root = Path(simg_root).resolve() if simg_root else None
        self.seed = int(seed)
        self.cache = read_graph_cache(graph_cache) if graph_cache else {}
        self.cache_path = str(Path(graph_cache).resolve()) if graph_cache else None
        self.cache_sha = sha256(graph_cache) if graph_cache else None
        self._pipeline = None
        self.provenance = {}

    def _load_upstream(self):
        if self._pipeline is not None:
            return
        if self.simg_root is not None:
            if not (self.simg_root / "simg/model_utils.py").is_file():
                raise ValueError("--simg-root must contain simg/model_utils.py")
            sys.path.insert(0, str(self.simg_root))
        else:
            bundled = Path(__file__).resolve().parents[1] / "vendor" / "simg_runtime"
            if (bundled / "simg/model_utils.py").is_file():
                sys.path.insert(0, str(bundled))
        # Upstream import prints status messages. Keep stdout reserved for JSON.
        with contextlib.redirect_stdout(sys.stderr):
            import simg.model_utils as upstream
        self._pipeline = upstream.pipeline
        self._upstream = upstream

    def __call__(self, smiles):
        key = canonical_smiles(smiles)
        if key in self.provenance:
            return classifier_graph(self.cache[key]), dict(self.provenance[key])
        if key in self.cache:
            return classifier_graph(self.cache[key]), {
                "schema": GRAPH_SCHEMA, "source": "trusted_simg_cache",
                "cache_sha256": self.cache_sha, "canonical_smiles": key}
        self._load_upstream()
        from simg.config import ATOMS
        mol = Chem.AddHs(Chem.MolFromSmiles(key))
        unknown = sorted({a.GetSymbol() for a in mol.GetAtoms()} - set(ATOMS))
        if unknown:
            raise ValueError("Unsupported elements in pretrained SIMG*: {}".format(unknown))
        params = AllChem.ETKDGv3()
        params.randomSeed = self.seed
        params.useRandomCoords = True
        params.maxAttempts = 1000
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError("3D conformer generation failed")
        if AllChem.MMFFHasAllMoleculeParams(mol):
            force_field = "MMFF94"
            convergence = AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
        elif AllChem.UFFHasAllMoleculeParams(mol):
            force_field = "UFF"
            convergence = AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
        else:
            raise ValueError("Neither MMFF94 nor UFF supports this molecule")
        if convergence != 0:
            raise ValueError("{} geometry optimization did not converge".format(force_field))
        obabel = shutil.which("obabel")
        if obabel is None:
            raise ValueError("Open Babel executable 'obabel' is required for the original SIMG connectivity protocol")
        coords = np.asarray(mol.GetConformer().GetPositions())
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        # Upstream parser is also used for provided SDF, avoiding shell-built
        # commands and temporary files in a user's current working directory.
        with tempfile.TemporaryDirectory(prefix="toxfm-") as temp:
            xyz, sdf = Path(temp) / "molecule.xyz", Path(temp) / "molecule.sdf"
            with xyz.open("w", encoding="ascii") as handle:
                handle.write(str(len(symbols)) + "\n\n")
                for symbol, point in zip(symbols, coords):
                    handle.write("{} {:.6f} {:.6f} {:.6f}\n".format(symbol, *point))
            subprocess.run([obabel, str(xyz), "-O", str(sdf)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            from simg.data import get_connectivity_info
            connectivity = get_connectivity_info(sdf_file=str(sdf))
            # Match the XYZ precision used to build the source graph cache.
            coords = np.round(coords, 6)
            with contextlib.redirect_stdout(sys.stderr), torch.no_grad():
                result = self._pipeline(symbols, coords, connectivity)
        graph = validate_graph(result[0])
        self.cache[key] = graph
        model_file = Path(self._upstream.__file__)
        provenance = {
            "schema": GRAPH_SCHEMA, "source": "simg_star_pipeline",
            "canonical_smiles": key, "conformer_seed": self.seed,
            "force_field": force_field, "geometry_converged": True,
            "connectivity": "OpenBabel from XYZ", "interaction_threshold": 0,
            "simg_model_utils_sha256": sha256(model_file),
            "lp_checkpoint_sha256": sha256(self._upstream.LP_CHECKPOINT_PATH),
            "nbo_checkpoint_sha256": sha256(self._upstream.INTERACTION_CHECKPOINT_PATH)}
        self.provenance[key] = provenance
        return classifier_graph(graph), provenance
