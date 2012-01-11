bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high-gamma', 'broad']
import operator as op

options = {'Avalanche.threshold_level': 10,
        'Avalanche.threshold_mode': 'Likelihood',
        'Avalanche.time_scale': (10.0, op.lt),
        'Filter.band_name': 'broad',
        'Filter.downsampled_rate': 1000,
        'Fit.distribution': 'power_law',
        'Fit.fixed_xmax': True,
        'Fit.fixed_xmin': None,
        'Fit.variable': 'size_events',
        'Fit.xmax': None,
        'Fit.xmin': 1,
        'Sensor.sensor_type': 'ECoG',
        'Subject.name': None,
        'Task.eyes': None}

params = [(b, dsr, S, v) for b in bands for dsr in [200, 1000] for S in ['A', 'K1'] for v in ['size_events']]

from alpha_sigma import alpha_sigma
for b, dsr, S, v in params:
    options['Filter.downsampled_rate'] = dsr
    options['Fit.variable'] = v
    options['Subject.name'] = S
    options['Filter.band_name'] = b
    alpha_sigma(options)
