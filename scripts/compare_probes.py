#!/usr/bin/env python3
"""Compare completed 1024- and 2048-world SuperMimic Manipulation probes.

Run from project root with: python scripts/compare_probes.py
No optional dependencies are required; optionally prints TensorBoard scalars if installed.
"""
import csv
import json
from pathlib import Path
from statistics import mean

BASE = Path('logs/supermimic')
NAMES = ('box_in_bin_probe_1024_100', 'box_in_bin_probe_2048_100')


def fmt(x, digits=2):
    return f'{x:,.{digits}f}'


def main():
    runs = []
    for name in NAMES:
        root = BASE / name
        result = json.loads((root / 'completion.json').read_text())
        meta = json.loads((root / 'params/runtime.json').read_text())
        runs.append((name, root, result, meta))
    a, b = runs
    for field in ('agent', 'run', 'task', 'seed', 'reset_bank', 'cuda_graph_mode'):
        if a[3].get(field) != b[3].get(field):
            print(f'WARNING: {field!r} differs; this is not a single-variable comparison')
    print(f"\nRun A: {a[0]}  |  Run B: {b[0]}")
    print('Note: throughput includes warmup; model updates scale with new data under epochs scheduling.')
    print(f"{'Metric':42s} {'1024':>16s} {'2048':>16s} {'ratio B/A':>12s}")
    keys = [
        ('real_transitions', 1),
        ('wall_seconds', 1),
        ('world_model_updates', 1),
        ('torch_peak_allocated_bytes', 1048576),
        ('replay_allocated_bytes', 1048576),
        ('surrogate_native_reward_mean_abs_gap', 1),
        ('surrogate_native_done_mismatch_rate', 1),
        ('candidate_surrogate_only_rate', 1),
        ('native_observation_max_error', 1),
        ('diverged_transitions', 1),
        ('physics_overflow_world_steps', 1),
        ('contact_match_overflow_world_steps', 1),
    ]
    for key, divisor in keys:
        x, y = a[2][key] / divisor, b[2][key] / divisor
        ratio = y / x if x else float('nan')
        print(f'{key:42s} {fmt(x, 4):>16s} {fmt(y, 4):>16s} {fmt(ratio, 3):>12s}')
    for label, func in (
        ('transitions_per_second', lambda r: r['real_transitions'] / r['wall_seconds']),
        ('wm_updates_per_second', lambda r: r['world_model_updates'] / r['wall_seconds']),
        ('seconds_per_iteration', lambda r: r['wall_seconds'] / r['iteration']),
    ):
        x, y = func(a[2]), func(b[2])
        print(f'{label:42s} {fmt(x, 3):>16s} {fmt(y, 3):>16s} {fmt(y / x, 3):>12s}')
    print('\n2048 GPU sampling (device-wide; includes other jobs if any):')
    csv_path = Path('logs/monitor') / b[0] / 'gpu.csv'
    if csv_path.exists():
        samples = []
        with csv_path.open(newline='') as f:
            for row in csv.reader(f):
                if len(row) != 6 or row[0] == 'date':
                    continue
                try:
                    samples.append((float(row[1]), float(row[3])))
                except ValueError:
                    continue
        if samples:
            print(f'  Samples: {len(samples)}')
            print(f'  GPU memory avg / max: {fmt(mean(x[0] for x in samples), 1)} / {fmt(max(x[0] for x in samples), 1)} MiB')
            print(f'  GPU utilization avg / max: {fmt(mean(x[1] for x in samples), 1)} / {fmt(max(x[1] for x in samples), 1)} %')
        else:
            print('  GPU CSV contains no valid samples')
    else:
        print('  Missing GPU CSV')

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print('\nTensorBoard Python package not available; skip scalar comparison.')
        return
    metrics = (
        'Model/wm_step_loss',
        'Model/wm_train_eval_step_loss',
        'Model/wm_val_step_loss',
        'Model/wm_val_recent_step_loss',
        'Model/wm_val_train_ratio',
        'Policy/policy_grad_norm',
        'Imagination/imagined_reward',
        'Model/candidate_surrogate_only_rate',
    )
    print('\nLast available TensorBoard scalar (not proof of convergence):')
    for k in metrics:
        vals = []
        for _, root, _, _ in runs:
            try:
                e = EventAccumulator(str(root / 'tensorboard'))
                e.Reload()
                scalars = e.Scalars(k) if k in e.Tags().get('scalars', []) else []
                vals.append(fmt(scalars[-1].value, 6) if scalars else 'N/A')
            except Exception:
                vals.append('N/A')
        print(f'  {k:42s} {vals[0]:>14s} {vals[1]:>14s}')


if __name__ == '__main__':
    main()
