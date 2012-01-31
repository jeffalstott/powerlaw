from distutils.core import setup
setup(
        name = 'powerlaw',
        py_modules = ['powerlaw'],
        version = '.1',
        description = 'Toolbox for testing if a probability distribution fits a power law',
        author='Jeff Alstott',
        author_email = 'jeffalstott@gmail.com',
        url = 'https://github.com/jeffalstott/',
        requires = ['scipy', 'numpy', 'matplotlib'],
        classifiers = [
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python',
            'Programming Language :: Python :: 2.7',
            'Operating System :: OS Independent',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Development Status :: 4 - Beta',
            'Intended Audience :: Science/Research'
            ]
        )
