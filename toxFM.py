#!/usr/bin/env python3
"""Predict stage-one blood toxicity and stage-two endpoint scores from SMILES."""
import argparse
import csv
import json
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("-InputSmiles", "--input-smiles", dest="smiles")
    source.add_argument("--input-csv", type=Path, help="CSV with a SMILES column")
    p.add_argument("-ToxcityType", "-ToxicityType", "--toxicity-type", default="all", dest="toxicity")
    p.add_argument("-InputGeneExpression", nargs="?", const="__provided__", default=None,
                   help="Unsupported: the manuscript model was not trained on gene expression")
    p.add_argument("--model-dir", type=Path, default=Path(__file__).resolve().parent / "model")
    p.add_argument("--simg-root", type=Path, default=None)
    p.add_argument("--graph-cache", type=Path, help="Optional trusted SIMG graph cache; never an RDKit substitute")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--conformer-seed", type=int, default=42)
    p.add_argument("--output", type=Path, help="New JSON file; existing files are never overwritten")
    args = p.parse_args(argv)
    if args.InputGeneExpression is not None:
        p.error("InputGeneExpression is not supported: no gene-expression encoder or trained multimodal weights exist in this manuscript method")
    if args.output and args.output.exists():
        p.error("--output already exists; choose a new filename")
    from toxfm.constants import SUBTYPES, endpoint_name
    from toxfm.features import SIMGFeaturizer
    from toxfm.io import write_json
    from toxfm.predict import TwoStagePredictor
    try:
        endpoint = endpoint_name(args.toxicity)
        if endpoint == "FirstStage":
            endpoints = ()
        else:
            endpoints = SUBTYPES if endpoint == "all" else (endpoint,)
        predictor = TwoStagePredictor(args.model_dir, endpoints, args.device)
        featurizer = SIMGFeaturizer(args.simg_root, args.graph_cache, args.conformer_seed)
        if args.smiles is not None:
            inputs = [{"SMILES": args.smiles}]
        else:
            with args.input_csv.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or "SMILES" not in reader.fieldnames:
                    raise ValueError("--input-csv requires a SMILES column")
                inputs = list(reader)
            if not inputs:
                raise ValueError("Input CSV is empty")
        predictions = []
        for row in inputs:
            graph, provenance = featurizer(row["SMILES"])
            result = predictor.predict(graph, provenance)
            if "ID" in row:
                result["id"] = row["ID"]
            predictions.append(result)
        payload = predictions[0] if args.smiles is not None else predictions
        if args.output:
            write_json(args.output, payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        p.exit(2, "toxFM: {}\n".format(exc))


if __name__ == "__main__":
    main()
