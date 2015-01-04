import unittest
import powerlaw
from numpy.testing import assert_allclose
from numpy import genfromtxt

references = {
        'words': {
            'discrete': True,
            'data': genfromtxt('reference_data/words.txt'),
            'alpha': 1.95,
            'xmin': 7,
            'lognormal': (0.395, 0.69),
            'exponential': (9.09, 0.0),
            'stretched_exponential': (4.13, 0.0),
            'truncated_power_law': (-0.899, 0.18),
            },
        'terrorism': {
            'discrete': True,
            'data': genfromtxt('reference_data/terrorism.txt'),
            'alpha': 2.4,
            'xmin': 12,
            'lognormal': (-0.278, 0.78),
            'exponential': (2.457, 0.01),
            'stretched_exponential': (0.772, 0.44),
            'truncated_power_law': (-0.077, 0.70),
            },
        'blackouts': {
            'discrete': False,
            'data': genfromtxt('reference_data/blackouts.txt')/10.0**3,
            'alpha': 2.3,
            'xmin': 230,
            'lognormal': (-0.412, 0.68),
            'exponential': (1.21, 0.23),
            'stretched_exponential': (-0.417, 0.68),
            'truncated_power_law': (-0.382, 0.38),
            },
        'cities': {
            'discrete': False,
            'data': genfromtxt('reference_data/cities.txt')/10**3,
            'alpha': 2.37,
            'xmin': 52.46,
            'lognormal': (-0.090, 0.93),
            'exponential': (3.65,0.0),
            'stretched_exponential': (0.204, 0.84),
            'truncated_power_law': (-0.123, 0.62),
            },
        'fires': {
            'discrete': False,
            'data': genfromtxt('reference_data/fires.txt'),
            'alpha': 2.2,
            'xmin': 6324,
            'lognormal': (-1.78, 0.08),
            'exponential': (4.00, 0.0),
            'stretched_exponential': (-1.82, 0.07),
            'truncated_power_law': (-5.02, 0.0),
            },
        'flares': {
            'discrete': False,
            'data': genfromtxt('reference_data/flares.txt'),
            'alpha': 1.79,
            'xmin': 323,
            'lognormal': (-0.803, 0.42),
            'exponential': (13.7, 0.0),
            'stretched_exponential': (-0.546, 0.59),
            'truncated_power_law': (-4.52, 0.0),
            },
        'quakes': {
            'discrete': False,
            'data': (10**genfromtxt('reference_data/quakes.txt'))/10**3,
            'alpha': 1.64,
            'xmin': .794,
            'lognormal': (-7.14, 0.0),
            'exponential': (11.6, 0.0),
            'stretched_exponential': (-7.09, 0.0),
            'truncated_power_law': (-24.4, 0.0),
            },
        'surnames': {
            'discrete': False,
            'data': genfromtxt('reference_data/surnames.txt')/10**3,
            'alpha': 2.5,
            'xmin': 111.92,
            'lognormal': (-0.836, 0.4),
            'exponential': (2.89, 0.0),
            'stretched_exponential': (-0.844, 0.40),
            'truncated_power_law': (-1.36, 0.10),
            }
        }

results = {
        'words': {},
        'terrorism': {},
        'blackouts': {},
        'cities': {},
        'fires': {},
        'flares': {},
        'quakes': {},
        'surnames': {}
        }

class FirstTestCase(unittest.TestCase):

    def test_power_law(self):
        print("Testing power law fits")

        rtol = .1
        atol = 0.01

        for k in references.keys():
            print(k)
            data = references[k]['data']
            fit = powerlaw.Fit(data, discrete=references[k]['discrete'])
            results[k]['alpha'] = fit.alpha
            results[k]['xmin'] = fit.xmin

            assert_allclose(fit.alpha, references[k]['alpha'],
                    rtol=rtol, atol=atol, err_msg=k)

            assert_allclose(fit.xmin, references[k]['xmin'],
                    rtol=rtol, atol=atol, err_msg=k)

    def test_lognormal(self):
        print("Testing lognormal fits")

        rtol = .1
        atol = 0.01

        for k in references.keys():
            print(k)
            data = references[k]['data']
            fit = powerlaw.Fit(data, discrete=references[k]['discrete'])

            Randp = fit.loglikelihood_ratio('power_law', 'lognormal',
                    normalized_ratio=True)
            results[k]['lognormal'] = Randp

            #assert_allclose(Randp, references[k]['lognormal'],
            #        rtol=rtol, atol=atol, err_msg=k)

    def test_exponential(self):
        print("Testing exponential fits")

        rtol = .1
        atol = 0.01

        for k in references.keys():
            print(k)
            data = references[k]['data']
            fit = powerlaw.Fit(data, discrete=references[k]['discrete'])

            Randp = fit.loglikelihood_ratio('power_law', 'exponential',
                    normalized_ratio=True)
            results[k]['exponential'] = Randp

            #assert_allclose(Randp, references[k]['exponential'],
            #        rtol=rtol, atol=atol, err_msg=k)

    def test_stretched_exponential(self):
        print("Testing stretched_exponential fits")

        rtol = .1
        atol = 0.01

        for k in references.keys():
            print(k)
            data = references[k]['data']
            fit = powerlaw.Fit(data, discrete=references[k]['discrete'])

            Randp = fit.loglikelihood_ratio('power_law', 'stretched_exponential',
                    normalized_ratio=True)
            results[k]['stretched_exponential'] = Randp

            #assert_allclose(Randp, references[k]['stretched_exponential'],
            #        rtol=rtol, atol=atol, err_msg=k)

    def test_truncated_power_law(self):
        print("Testing truncated_power_law fits")

        rtol = .1
        atol = 0.01

        for k in references.keys():
            print(k)
            if references[k]['discrete']:
                continue
            data = references[k]['data']
            fit = powerlaw.Fit(data, discrete=references[k]['discrete'])

            Randp = fit.loglikelihood_ratio('power_law', 'truncated_power_law')
            results[k]['truncated_power_law'] = Randp

            #assert_allclose(Randp, references[k]['truncated_power_law'],
            #        rtol=rtol, atol=atol, err_msg=k)

if __name__ == '__main__':
    # execute all TestCases in the module
    unittest.main()
