# toxFM: SIMG*–ParseCaps for Hematological Toxicity Prediction

Predict blood toxicity and four hematological toxicity endpoints from molecular SMILES.

## Environment Setup

Main requirements:

- Python 3.8
- PyTorch 1.11.0
- PyTorch Geometric 2.0.4
- RDKit 2021.03.2
- Open Babel 3.1.0
- PyTorch Lightning 1.7.7

Install on Linux:

```bash
conda env create -f environment.yml
conda activate toxfm38
bash scripts/install_torch_pyg.sh cu113
python -m pip install -r requirements.txt
```

For CPU-only installation, replace `cu113` with `cpu`.

## Quick Start

- `data/`: five toxicity datasets with `SMILES` and `Label` columns.
- `model/`: trained models.
- `configs/selected_method.json`: model and training configurations.

### Prediction

```bash
python toxFM.py -InputSmiles 'CCO' -ToxcityType Myelotoxicity
python toxFM.py -InputSmiles 'CCO' -ToxcityType all --output outputs/prediction.json
python toxFM.py --input-csv molecules.csv -ToxcityType all --output outputs/batch.json
```

Input CSV requires a `SMILES` column. Available endpoints: `Hemolytic`,
`Hematotoxicity`, `Myelotoxicity`, `Thrombocytopenia`, or `all`.
Output includes the stage-one blood-toxicity score and requested stage-two scores.

### Feature Extraction

```bash
python prepare_graphs.py --data-root data --output outputs/graphs.pt --allow-failures
```

`--allow-failures` saves successfully processed molecules and records failed inputs.

### Train and Test

```bash
python train.py --data-root data --graph-cache outputs/graphs.pt \
  --output-dir runs/repeat_01 --epochs 160 --patience 25 --device cuda
```

Training uses five-fold scaffold splits. Use a new output directory for each run.

### Prediction with Your Trained Models

```bash
python toxFM.py -InputSmiles 'CCO' -ToxcityType all --model-dir runs/repeat_01
```

## Acknowledgments

Based on [SIMG](https://github.com/gomesgroup/simg) and inspired by
[ParseCaps](https://github.com/ornamentt/Parsecaps).
Upstream licenses are retained in `vendor/simg_runtime/LICENSE` and `vendor/ParseCaps_LICENSE`.
