"""Read-only final-weight health sweep; finite budget failures remain failures."""
import time

import numpy as np
import torch

from ..h2pro_snn.stream_data import prefetch, stream_batches
from ..h2pro_snn.training import cpu_weights, memory_stats
from .monitor import HealthMonitor
from .training import forward_batch


@torch.no_grad()
def frozen_train_sweep(model, store, device, config):
    if not model.analog:
        raise ValueError("This sweep reports continuous ANN activation units")
    before = cpu_weights(model)
    monitor = HealthMonitor(config['stability_limits'], config['warning_limits'])
    state = None
    failures = []
    events = windows = candidates = above80 = background_above80 = 0
    minimum, maximum = np.inf, -np.inf
    started = time.perf_counter()
    for index, (batch, _) in enumerate(prefetch(iter(stream_batches(
            store, 'train_main', config['batch_size'], seed=config['seed'], epoch=1))), 1):
        logits, state, _, _ = forward_batch(model, batch, state, device)
        values = logits.cpu().numpy()[batch['candidate']]
        labels = batch['labels'][batch['candidate']]
        # Count before guard enforcement so finite outlier batches are included.
        candidates += len(values)
        if len(values):
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
            above80 += int((np.abs(values) > 80).sum())
            background_above80 += int(((values > 80) & (labels == 0)).sum())
        try:
            monitor.observe(model, logits, batch['candidate'], batch['active'])
        except FloatingPointError as error:
            if ' exceeds ' not in str(error):
                raise
            failures.append({'batch': index, 'error': str(error),
                             'refs': [{k: v for k, v in ref.items() if k != 'ids'} for ref in batch['refs']]})
        events += len(batch['labels']); windows += int(batch['active'].sum())
        if index % 500 == 0:
            print({'stage': 'frozen_ann_train_main', 'batches': index, 'events': events,
                   'outlier_batches': len(failures)}, flush=True)
    expected = store.manifest['roles']['train_main']
    assert events == sum(r['published_events'] for r in expected)
    assert windows == sum(r['closed_windows'] for r in expected)
    assert candidates == sum(r['candidate_events'] for r in expected)
    assert all(torch.equal(v, cpu_weights(model)[k]) for k, v in before.items())
    return {'status': 'passed' if not failures else 'finite_amplitude_budget_outliers',
            'role': 'train_main', 'events': events, 'windows': windows, 'batches': index,
            'outlier_batches': failures, 'stability': monitor.summary(),
            'all_candidate_logits_including_outliers': {'count': candidates, 'min': minimum, 'max': maximum,
                'abs_above80': above80, 'background_above80': background_above80},
            'monitor_note': 'Layer maxima include outliers. Only the separate all_candidate_logits field includes logits of guard-outlier batches.',
            'activation_units': 'Continuous charge/theta; not spikes or AC. Same amplitude budget as v2.2.',
            'parameter_updates': 0, 'parameters_unchanged': True, 'seconds': time.perf_counter()-started,
            **memory_stats(device)}
