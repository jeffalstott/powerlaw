"""
Methods for loading test datasets.

For information on test datasets, including sources (where available)
see the documentation.

One dataset, ``weblinks.hist`` is given as a histogram, not sampled data
so we can't analyze it with this library.
"""

import numpy as np
import os
import pathlib

from .plotting import plot_pdf

# The test datasets should be placed in a subdirectory called 'reference_data'
# in the same directory this file.
"""
@private
No need to show this variable in the documentation as it will
vary from system to system.
"""
TEST_DATASET_DIR = os.path.join((pathlib.Path(__file__).parent.resolve()), 'reference_data')


TEST_DATASETS = {
                'blackouts': 'Number of people in the United States affected by electricity blackouts between 1984 and 2002.',
                'cities': '',
                'fires': '',
                'flares': '',
                'quakes': '',
                'surnames': '',
                'terrorism': '',
                'words': 'Frequency of word usage in Herman Melville’s novel "Moby Dick"',
                }
"""
Dictionary of all test datasets with a brief description.
"""


def load_test_dataset(name):
    """
    Load any of the test datasets.

    See ``print_test_datasets()`` for available datasets and brief
    descriptions.

    Parameters
    ----------
    name : str
        The name of a test dataset; see `print_test_datasets()`.

    Returns
    -------
    data : numpy.ndarray
        The sampled values from the test dataset.
    """
    assert name in TEST_DATASETS.keys(), \
            f'Dataset \'{name}\' not recognized; see \'print_test_datasets()\' for available options.'

    data = np.genfromtxt(os.path.join(TEST_DATASET_DIR, f'{name}.txt'))

    return data


def print_test_datasets():
    """
    Print out the available test datasets.
    """
    print('**Available datasets**')
    print(f'Data location: {TEST_DATASET_DIR}\n')

    maxNameLength = np.max([len(k) for k in TEST_DATASETS.keys()])

    print('Datasets:')
    for k, v in TEST_DATASETS.items():
        print(f'{k}{"."*(maxNameLength+5-len(k))}{v}')


def plot_test_datasets():
    """
    Show the PDF for all of the available test datasets.
    """
    import matplotlib.pyplot as plt

    dataset_names = list(TEST_DATASETS.keys())
    fig = plt.figure(figsize=(len(dataset_names)*3, 3.5))

    for i in range(len(dataset_names)):
        ax = fig.add_subplot(1, len(dataset_names), i+1)

        data = load_test_dataset(dataset_names[i])
        
        plot_pdf(data, ax=ax)

        ax.set_title(f'{dataset_names[i]}\n$N = {len(data)}$')

    fig.tight_layout()
    plt.show()
