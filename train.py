#!/usr/bin/env python3
"""Train the selected SIMG*-ParseCaps method with scaffold validation.

Stage 1 and the four stage-2 heads are independently supervised classifiers.
Keep the exact early-stopped weights used to select each threshold: no refit.
Every output directory must be new; original data and checkpoints are read-only.
"""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from toxfm.constants import ENDPOINT_FILES, GRAPH_SCHEMA
from toxfm.features import read_graph_cache, classifier_graph
from toxfm.io import sha256, write_json
from toxfm.loss import make_criterion
from toxfm.metrics import evaluate, choose_threshold, METRIC_NAMES
from toxfm.model import ARCHITECTURE, FormalSIMGHouseholderParseCaps
from toxfm.preprocessing import (make_endpoint_dataset, build_descriptor_matrix,
                                fit_descriptor_scaler, transform_descriptors,
                                prepare_graph_subset)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def assert_disjoint(records, left, right):
    if set(left) & set(right):
        raise ValueError('Sample split overlap')
    if {records[i]['scaffold'] for i in left} & {records[i]['scaffold'] for i in right}:
        raise ValueError('Scaffold split overlap')


@torch.no_grad()
def predict(model, data, device, batch_size):
    model.eval()
    labels, scores = [], []
    for batch in DataLoader(data, batch_size=batch_size, shuffle=False):
        batch = batch.to(device)
        labels.extend(batch.y.view(-1).cpu().tolist())
        scores.extend(model(batch)[1].cpu().tolist())
    return np.asarray(labels), np.asarray(scores)


def fit(train_data, validation_data, model_config, training_config, seed, args, device):
    seed_everything(seed)
    model = FormalSIMGHouseholderParseCaps(**model_config).to(device)
    labels = np.asarray([g.y.item() for g in train_data], dtype=int)
    criterion, loss_config = make_criterion(training_config['loss_name'], labels)
    criterion = criterion.to(device)
    if training_config['sampler'] == 'class_balanced':
        counts = np.bincount(labels, minlength=2)
        if not (counts > 0).all():
            raise ValueError('Both training classes required')
        weights = torch.tensor(1.0 / counts[labels], dtype=torch.double)
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generator)
        loader = DataLoader(train_data, batch_size=args.batch_size, sampler=sampler)
    elif training_config['sampler'] == 'standard':
        loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    else:
        raise ValueError('Unsupported sampler')
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'],
                                  weight_decay=training_config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best, best_state, stale, history = None, None, 0, []
    metric = 'AUPR' if args.current_endpoint in ('Myelotoxicity', 'Thrombocytopenia') else 'AUCROC'
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch)[0], batch.y.view(-1))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        y, scores = predict(model, validation_data, device, args.batch_size)
        threshold = choose_threshold(y, scores)
        validation = evaluate(y, scores, threshold)
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), **validation)
        history.append(row)
        if best is None or row[metric] > best[metric] + 1e-8:
            best = row
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(best_state, strict=True)
    model.eval()
    # Threshold belongs to these exact weights, not a separately refit model.
    y, scores = predict(model, validation_data, device, args.batch_size)
    threshold = choose_threshold(y, scores)
    return model, best_state, threshold, best, loss_config, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--graph-cache', type=Path, required=True, help='Trusted SIMG* cache only')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).parent / 'configs/selected_method.json')
    p.add_argument('--endpoints', default='all', help='Comma-separated canonical names or all')
    p.add_argument('--epochs', type=int, default=160)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=20260910)
    p.add_argument('--repeat-index', type=int, default=0)
    p.add_argument('--outer-fold', type=int, default=None, choices=range(1, 6))
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        p.error('epochs, patience and batch-size must be positive')
    if args.output_dir.exists():
        p.error('output-dir already exists; select a new directory')
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable; use a Slurm GPU job or explicitly --device cpu')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    endpoints = list(ENDPOINT_FILES) if args.endpoints == 'all' else args.endpoints.split(',')
    if not endpoints or set(endpoints) - set(ENDPOINT_FILES):
        p.error('Unknown endpoints')
    device = torch.device(args.device)
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'run_config.json', {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'selected_method_sha256': sha256(args.config), 'graph_cache_sha256': sha256(args.graph_cache),
        'protocol': 'Fixed development-selected configurations; inner scaffold early stopping; same-model thresholds; no refit',
    })
    graph_map = read_graph_cache(args.graph_cache)
    bundle = dict(format_version=1, architecture=ARCHITECTURE, graph_schema=GRAPH_SCHEMA,
                  bundle_name=args.output_dir.name, endpoints={})
    metric_rows = []
    for endpoint in endpoints:
        args.current_endpoint = endpoint
        records, audit = make_endpoint_dataset(args.data_root, endpoint, graph_map)
        if len(records) < 10:
            raise ValueError('Too few matched molecules for ' + endpoint)
        task_dir = args.output_dir / endpoint
        task_dir.mkdir()
        write_json(task_dir / 'data_audit.json', audit)
        pd.DataFrame(records).to_csv(task_dir / 'sample_manifest.csv', index=False)
        graph_list = [classifier_graph(graph_map[r['canonical_smiles']], r['label']) for r in records]
        descriptors = build_descriptor_matrix(records)
        labels = np.asarray([r['label'] for r in records])
        groups = np.asarray([r['scaffold'] for r in records])
        repeat_seed = args.seed + args.repeat_index
        splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=repeat_seed).split(
            np.zeros(len(labels)), labels, groups))
        test_coverage = np.concatenate([test for _, test in splits])
        if len(test_coverage) != len(records) or len(set(test_coverage)) != len(records):
            raise RuntimeError('Outer folds must cover each sample exactly once')
        bundle['endpoints'][endpoint] = []
        for fold, (outer_train, test) in enumerate(splits, 1):
            if args.outer_fold and fold != args.outer_fold:
                continue
            if len(np.unique(labels[test])) != 2:
                raise ValueError('Outer test fold has one class; choose another seed before training')
            inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=repeat_seed + 1000 + fold)
            inner_split = next(((tr, va) for tr, va in inner.split(
                np.zeros(len(outer_train)), labels[outer_train], groups[outer_train])
                if len(np.unique(labels[outer_train[tr]])) == 2 and len(np.unique(labels[outer_train[va]])) == 2), None)
            if inner_split is None:
                raise ValueError('Cannot create two-class inner scaffold validation')
            train, validation = outer_train[inner_split[0]], outer_train[inner_split[1]]
            for left, right in ((train, validation), (train, test), (validation, test)):
                assert_disjoint(records, left, right)
            selected = next(c for c in config['tasks'][endpoint] if c['fold'] == fold)
            if 'profile' in selected:
                selected = config['profiles'][selected['profile']]
            scaler = fit_descriptor_scaler(descriptors, train) if selected['model_config']['descriptor_dim'] else None
            scaled = transform_descriptors(descriptors, scaler) if scaler else None
            train_data, val_data, test_data = [prepare_graph_subset(graph_list, indices, scaled)
                                              for indices in (train, validation, test)]
            model, state, threshold, best, loss_config, history = fit(
                train_data, val_data, selected['model_config'], selected['training_config'],
                repeat_seed + fold * 100, args, device)
            y, scores = predict(model, test_data, device, args.batch_size)
            metrics = dict(endpoint=endpoint, fold=fold, repeat=args.repeat_index + 1,
                           **evaluate(y, scores, threshold))
            metric_rows.append(metrics)
            fold_dir = task_dir / ('fold_%02d' % fold)
            fold_dir.mkdir()
            write_json(fold_dir / 'split.json', {'train_indices': train.tolist(),
                       'validation_indices': validation.tolist(), 'test_indices': test.tolist()})
            clean_scaler = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in scaler.items()} if scaler else None
            filename = '%s/fold_%02d/model.pt' % (endpoint, fold)
            torch.save(dict(architecture=ARCHITECTURE, graph_schema=GRAPH_SCHEMA,
                            endpoint=endpoint, fold=fold, model_config=selected['model_config'],
                            model_state=state, threshold=threshold, descriptor_scaler=clean_scaler,
                            training_config=selected['training_config'], loss_config=loss_config,
                            selection=best, seed=repeat_seed + fold * 100,
                            provenance={'threshold_scope': 'same_inner_validation_model_no_refit',
                                        'data_csv_sha256': sha256(args.data_root / ENDPOINT_FILES[endpoint])}),
                       args.output_dir / filename)
            bundle['endpoints'][endpoint].append({'file': filename,
                'sha256': sha256(args.output_dir / filename), 'threshold': threshold})
            pd.DataFrame(history).to_csv(fold_dir / 'training_history.csv', index=False)
            pd.DataFrame({'row_index': [records[i]['row_index'] for i in test], 'label': y,
                          'score': scores, 'threshold': threshold}).to_csv(fold_dir / 'outer_predictions.csv', index=False)
            print(endpoint, fold, json.dumps(metrics), flush=True)
            del model
            torch.cuda.empty_cache()
    frame = pd.DataFrame(metric_rows)
    frame.to_csv(args.output_dir / 'fold_metrics.csv', index=False)
    summary = frame.groupby('endpoint')[list(METRIC_NAMES)].agg(['mean', 'std'])
    summary.columns = ['%s_%s' % c for c in summary.columns]
    summary.to_csv(args.output_dir / 'summary_mean_sd.csv')
    write_json(args.output_dir / 'manifest.json', bundle)


if __name__ == '__main__':
    main()
